"""Same frozen inverse likelihood and causal belief, different policy interfaces."""
from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import reacher_information_protocol_v37 as p
from jax_experiments.analysis import run_reacher_causal_v35 as source
from jax_experiments.analysis import run_reacher_confirmation_v36 as confirmation

common = source.common


def conditioning(arm, *, protocol=p):
    if arm not in protocol.CONDITIONED_ARMS:
        return None
    # The frozen inverse stays on the sampler; no per-step GPU/CPU transfers.
    with jax.default_device(jax.devices("cpu")[0]):
        gains = jnp.asarray(((1, 1), (-1, 1), (1, -1), (-1, -1)), jnp.float32)
        evidence = source.load_evidence(protocol.PRIMARY_MODEL, gains)
    return evidence, protocol.CONTROLS, arm != "conditioned_no_filter"


def train(arm, seed, smoke=False, *, protocol=p):
    return common.train_policy(
        "reacher", arm, seed, smoke=smoke, sampling_device="cpu",
        run_root=protocol.RUN_ROOT / "smoke" if smoke else protocol.RUN_ROOT,
        bundle_root=protocol.BUNDLE_ROOT, protocol_name=protocol.PROTOCOL,
        conditioning=conditioning(arm, protocol=protocol))


def validate_budget(metadata, arm, seed, *, protocol=p):
    if (metadata["protocol"] != protocol.PROTOCOL or metadata["seed"] != seed or metadata["arm"] != arm
            or metadata["runtime_versions"] != protocol.RUNTIME or metadata["sampling_device"] != "cpu"
            or (metadata["next_iteration"], metadata["total_steps"], metadata["update_count"])
            != (400, 1600000, 100000)):
        raise ValueError("information-matched controller budget/runtime differs")
    if metadata["policy_input_dim"] != (15 if arm in protocol.CONDITIONED_ARMS else 11):
        raise ValueError("controller observation does not match its arm")
    if arm in protocol.CONDITIONED_ARMS and (metadata["belief_controls"] != list(protocol.CONTROLS)
            or metadata["temporal_filter"] != (arm == "conditioned")):
        raise ValueError("training/evaluation belief definitions differ")


def audit(seed, *, protocol=p):
    for arm in protocol.ARMS:
        validate_budget(json.loads((protocol.screen_bundle(arm, seed) / "bundle.json").read_text()),
                        arm, seed, protocol=protocol)
    source.audit(seed, protocol=protocol, conditioned_arms=protocol.CONDITIONED_ARMS)


def smoke(*, protocol=p):
    if jax.default_backend() != "gpu":
        raise ValueError("smoke requires a GPU learner")
    runs = [train(arm, protocol.POLICY_SEEDS[0], smoke=True, protocol=protocol)
            for arm in protocol.CONDITIONED_ARMS]
    cpu = jax.devices("cpu")[0]
    timings = {}
    with jax.default_device(cpu):
        world = common.World("reacher", protocol.POLICY_SEEDS[0])
        agent = common.SACBase(15, 2, common.config("reacher", protocol.POLICY_SEEDS[0]), protocol.POLICY_SEEDS[0])
        reset_key, rollout_key = jax.random.split(jax.random.PRNGKey(protocol.POLICY_SEEDS[0]))
        inputs = (nnx.state(agent.policy, nnx.Param), world.reset(world.sys, reset_key), jnp.full(4, .25),
                  jnp.asarray(0), jax.random.split(rollout_key, common.p.SAMPLES),
                  jnp.asarray(common.training_modes(protocol.POLICY_SEEDS[0], common.p.SAMPLES, world.dwell)))
        for arm in protocol.CONDITIONED_ARMS:
            evidence, controls, temporal = conditioning(arm, protocol=protocol)
            scan = common.build_training_scan(world, nnx.graphdef(agent.policy), evidence, True,
                                              controls=controls, temporal=temporal)
            for attempt in range(2):
                started = time.perf_counter()
                result = scan(*inputs)
                jax.block_until_ready(result)
                seconds = time.perf_counter() - started
                print(f"SMOKE_ROLLOUT arm={arm} samples=4000 compile={attempt == 0} seconds={seconds:.3f}", flush=True)
            if not all(np.isfinite(np.asarray(value)).all() for value in result[1]):
                raise ValueError("nonfinite conditioned full rollout")
            if seconds > 120:
                raise ValueError("online inference sampler needs optimization before full training")
            timings[arm] = {"steady_4000_step_seconds": seconds,
                            "projected_400_iteration_rollout_seconds": seconds * 400}
    common.write_json(protocol.SMOKE, {"passed": True, "runtime_versions": source.require_runtime(),
                                     "primary_model_seed": protocol.PRIMARY_MODEL, "runs": runs, "sampling": timings})


def summarize(*, protocol=p):
    rows = [json.loads(protocol.audit_path(seed).read_text()) for seed in protocol.POLICY_SEEDS]
    for seed, row in zip(protocol.POLICY_SEEDS, rows, strict=True):
        if row["seed"] != seed or row["runtime_versions"] != protocol.RUNTIME or set(row["budgets"]) != set(protocol.ARMS):
            raise ValueError("information-matched audit identity differs")
        for arm in protocol.ARMS:
            validate_budget(row["budgets"][arm], arm, seed, protocol=protocol)
    primary = f"causal_{protocol.PRIMARY_MODEL}"
    means = {arm: [row["means"][arm] for row in rows] for arm in rows[0]["means"]}
    pairs = ((primary, "conditioned"), ("no_filter", "conditioned_no_filter"),
             (primary, "robust"), ("no_filter", "robust"),
             ("conditioned", "robust"), ("conditioned_no_filter", "robust"))
    comparisons = {f"{left}-minus-{right}": common.paired_summary(means[left], means[right]) for left, right in pairs}
    valid = all(e["steps"] == 200 and e["switches_observed"] == 3 and not e["termination"]
                for row in rows for episodes in row["events"].values() for e in episodes)
    if not valid:
        raise ValueError("information-matched evaluation lost its full switching windows")
    model = confirmation.frozen_model_metadata()
    provider = json.loads(protocol.FROZEN_METADATA.read_text())["provider"]
    return {
        "protocol": protocol.PROTOCOL, "policy_seeds": list(protocol.POLICY_SEEDS), "primary_model_seed": protocol.PRIMARY_MODEL,
        "means": {arm: float(np.mean(values)) for arm, values in means.items()}, "seed_means": means,
        "comparisons": comparisons, "valid_switch_windows": valid,
        "stationary_means": {arm: float(np.mean([e["return"] for row in rows for e in row["stationary"][arm]]))
                             for arm in rows[0]["stationary"]},
        "policy_parameters": {arm: [row["budgets"][arm]["policy_parameters"] for row in rows] for arm in protocol.ARMS},
        "same_information": {"inverse_model_seed": protocol.PRIMARY_MODEL, "filter_controls": list(protocol.CONTROLS),
                             "action_uses": "current observation and pre-action belief only",
                             "belief_updated_after_completed_transition": True},
        "shared_estimator_cost": {"provider_interactions": provider["total_steps"],
                                  "provider_updates": provider["update_count"],
                                  "calibration_interactions": model["calibration_interactions"],
                                  "estimator_updates": model["updates"], "retrained": False,
                                  "charged_to_both_same_information_routes": True,
                                  "charged_to_memoryless_robust": False},
    }


def analyze():
    result = summarize()
    primary_comparison = result["comparisons"][f"causal_{p.PRIMARY_MODEL}-minus-conditioned"]
    result.update({
        "development_evidence": {"positive_primary_paired_ci": primary_comparison["ci95"][0] > 0,
                                 "primary_wins": primary_comparison["wins"]},
        "inference_unit": "three fresh development policy seeds; episodes are not independent seeds",
        "next": "fresh confirmation if the matched comparison is positive; otherwise retain the inconclusive or negative baseline result",
    })
    common.write_json(p.ANALYSIS, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train", "audit", "analyze"))
    parser.add_argument("--seed", type=int, choices=p.POLICY_SEEDS)
    parser.add_argument("--arm", choices=p.ARMS)
    args = parser.parse_args()
    source.require_runtime()
    confirmation.frozen_model_metadata()
    if args.mode in ("train", "audit") and args.seed is None:
        parser.error("train/audit require a preregistered development seed")
    if args.mode == "train":
        if args.arm is None:
            parser.error("train requires an arm")
        if jax.default_backend() != "gpu":
            raise ValueError("training requires GPU updates")
        train(args.arm, args.seed)
    elif args.mode == "audit":
        audit(args.seed)
    else:
        {"smoke": smoke, "analyze": analyze}[args.mode]()
    print(f"REACHER_INFORMATION_COMPLETE mode={args.mode}", flush=True)


if __name__ == "__main__":
    main()
