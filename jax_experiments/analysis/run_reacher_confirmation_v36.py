"""Reuse the admitted trainer and frozen estimator; no new algorithm tuning."""
from __future__ import annotations

import argparse
import json
import statistics

import jax
import jax.numpy as jnp
import numpy as np

from jax_experiments.analysis import reacher_confirmation_protocol_v36 as p
from jax_experiments.analysis import run_reacher_causal_v35 as source

common = source.common


def frozen_model_metadata():
    metadata = json.loads(p.FROZEN_METADATA.read_text())["model"]
    if (metadata["model_seed"] != p.PRIMARY_MODEL or metadata["runtime_versions"] != p.RUNTIME
            or metadata["filter_controls"] != list(p.CONTROLS)
            or metadata["updates"] != p.source.MODEL_UPDATES):
        raise ValueError("frozen estimator differs from the preregistered model")
    if metadata["provider_seed"] in p.POLICY_SEEDS:
        raise ValueError("confirmation includes the calibration provider")
    return metadata


def train(arm, seed, smoke=False):
    return common.train_policy(
        "reacher", arm, seed, smoke=smoke, sampling_device="cpu",
        run_root=p.RUN_ROOT / "smoke" if smoke else p.RUN_ROOT,
        bundle_root=p.BUNDLE_ROOT, protocol_name=p.PROTOCOL)


def smoke():
    if jax.default_backend() != "gpu":
        raise ValueError("smoke requires GPU updates")
    learner = train("reference", p.POLICY_SEEDS[0], smoke=True)
    world = common.World("reacher", p.POLICY_SEEDS[0])
    state = world.reset(world.sys, jax.random.PRNGKey(7))
    evidence = source.load_evidence(p.PRIMARY_MODEL, world.gains)
    likelihood = np.asarray(evidence(state.obs, jnp.zeros(world.act_dim), state.obs))
    if not np.isfinite(likelihood).all():
        raise ValueError("nonfinite frozen-estimator smoke")
    common.write_json(p.SMOKE, {
        "passed": True, "runtime_versions": source.require_runtime(),
        "primary_model_seed": p.PRIMARY_MODEL, "filter_controls": list(p.CONTROLS),
        "learner": learner, "frozen_likelihood": likelihood.tolist()})


def analyze():
    rows = [json.loads(p.audit_path(seed).read_text()) for seed in p.POLICY_SEEDS]
    for row, seed in zip(rows, p.POLICY_SEEDS, strict=True):
        if row["seed"] != seed or row["runtime_versions"] != p.RUNTIME:
            raise ValueError("confirmation audit identity/runtime differs")
        for arm, metadata in row["budgets"].items():
            if (metadata["protocol"] != p.PROTOCOL or metadata["seed"] != seed
                    or metadata["arm"] != arm or metadata["sampling_device"] != "cpu"
                    or metadata["runtime_versions"] != p.RUNTIME
                    or (metadata["next_iteration"], metadata["total_steps"], metadata["update_count"])
                    != (400, 1600000, 100000)):
                raise ValueError("confirmation controller budget differs")
    source.analyze(protocol=p)
    result = json.loads(p.ANALYSIS.read_text())
    result["cohort"] = "five fresh policy seeds; V34/V35 exploratory seeds excluded"
    result["identification"] = {}
    for arm in (f"causal_{p.PRIMARY_MODEL}", "no_filter", "single_head"):
        episodes = [e for row in rows for e in row["events"][arm]]
        delays = [delay for e in episodes for delay in e["delays"]]
        result["identification"][arm] = {
            "mean_mode_accuracy": statistics.mean(e["mode_accuracy"] for e in episodes),
            "mean_brier": statistics.mean(e["brier"] for e in episodes),
            "switches": len(delays), "detected": sum(d["delay"] is not None for d in delays),
            "median_delay_including_undetected": statistics.median(
                d["delay"] if d["delay"] is not None else d["window_steps"] for d in delays),
            "mean_action_error_energy": statistics.mean(e["action_error_energy"] for e in episodes),
            "switch_window_mean_reward_sum": {
                str(width): statistics.mean(e["switch_windows"][str(width)]["reward_sum"] for e in episodes)
                for width in (1, 2, 4, 8, 16, 32)},
        }
    model = frozen_model_metadata()
    provider = json.loads(p.FROZEN_METADATA.read_text())["provider"]
    result["shared_estimator_cost"] = {
        "provider_seed": provider["seed"], "provider_interactions": provider["total_steps"],
        "provider_updates": provider["update_count"],
        "calibration_interactions": model["calibration_interactions"],
        "estimator_updates": model["updates"], "estimator_parameters": model["parameters"],
        "retrained_in_confirmation": False,
        "equal_controller_budget": True, "equal_total_training_cost": False,
    }
    common.write_json(p.ANALYSIS, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train", "audit", "analyze"))
    parser.add_argument("--seed", type=int, choices=p.POLICY_SEEDS)
    parser.add_argument("--arm", choices=("reference", "robust"))
    args = parser.parse_args()
    source.require_runtime()
    frozen_model_metadata()
    if args.mode in ("train", "audit") and args.seed is None:
        parser.error("train/audit require a fresh preregistered seed")
    if args.mode == "train":
        if args.arm is None:
            parser.error("train requires an arm")
        if jax.default_backend() != "gpu":
            raise ValueError("training requires GPU updates")
        train(args.arm, args.seed)
    elif args.mode == "audit":
        source.audit(args.seed, protocol=p)
    else:
        {"smoke": smoke, "analyze": analyze}[args.mode]()
    print(f"REACHER_CONFIRMATION_COMPLETE mode={args.mode}", flush=True)


if __name__ == "__main__":
    main()
