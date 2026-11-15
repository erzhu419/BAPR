"""Frozen independent-environment oracle-headroom screen for Swimmer."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import final_task_sweep
from jax_experiments.common.checkpoint import (
    _patch_flax_variablestate_unpickle,
    _restore_tree_like,
    _to_numpy_tree,
    load_checkpoint,
)
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.envs.stochastic_mode_env import StochasticModeEnv
from jax_experiments.train import make_algo, make_env


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (91_003, 91_021, 91_039)
EVENT_SEEDS = (199_201, 199_217, 199_233)
SCHEDULES = {
    199_201: (0, 1, 2, 3),
    199_217: (3, 0, 1, 2),
    199_233: (1, 3, 2, 0),
}
ARMS = ("robust", "reference")
ROUTES = ("robust", "uncompensated", "oracle", "native_reference")
RUN_ROOT = ROOT / "jax_experiments/results_swimmer_polarity_screen_v1"
BUNDLE_ROOT = ROOT / "jax_experiments/eval_bundles_swimmer_polarity_screen_v1"
AUDIT_ROOT = RUN_ROOT / "audits"
ANALYSIS_PATH = RUN_ROOT / "analysis" / "analysis.json"
MAX_ITERS = 2100
TOTAL_STEPS = 8_400_000
UPDATES = 525_000
DWELL = 250
HORIZON = 1000
EPISODES = 5


def run_dir(arm: str, seed: int) -> Path:
    return RUN_ROOT / arm / f"seed_{seed}"


def bundle_dir(arm: str, seed: int) -> Path:
    return BUNDLE_ROOT / arm / f"seed_{seed}"


def train_command(arm: str, seed: int) -> list[str]:
    directory = run_dir(arm, seed)
    cmd = [
        sys.executable, "-u", "-m", "jax_experiments.train",
        "--algo", "sac", "--env", "Swimmer-v2", "--seed", str(seed),
        "--max_iters", str(MAX_ITERS),
        "--save_root", str(directory.parent), "--run_name", directory.name,
        "--env_type", "stochastic_mode",
        "--stochastic_mode_family", "actuator_polarity_2d",
        "--stochastic_mode_dwell_steps", str(DWELL),
        "--stochastic_mode_dwell_distribution", "fixed",
        "--task_num", "4", "--test_task_num", "4",
        "--samples_per_iter", "4000", "--updates_per_iter", "250",
        "--start_train_steps", "4000", "--initial_random_steps", "0",
        "--ensemble_size", "2", "--hidden_dim", "256",
        "--lr", "0.0003", "--max_episode_steps", str(HORIZON),
        "--backend", "generalized", "--eval_protocol", "stationary",
        "--log_interval", "50", "--eval_episodes", "3",
        "--save_interval", "50", "--resume",
    ]
    if arm == "reference":
        cmd.extend(("--stochastic_mode_fixed_id", "0"))
    return cmd


def train(arm: str, seed: int) -> None:
    destination = bundle_dir(arm, seed)
    manifest = destination / "bundle.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text())
        if (payload.get("arm"), payload.get("seed")) != (arm, seed):
            raise ValueError(f"unexpected existing policy bundle: {destination}")
        if not (destination / "policy_params.pkl").is_file():
            raise ValueError(f"missing policy in existing bundle: {destination}")
        return
    command = train_command(arm, seed)
    print("TRAIN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)

    directory = run_dir(arm, seed)
    config = final_task_sweep.load_config(directory)
    if (config.env_name != "Swimmer-v2"
            or config.brax_backend != "generalized"
            or config.stochastic_mode_family != "actuator_polarity_2d"
            or config.stochastic_mode_fixed_id != (0 if arm == "reference" else -1)):
        raise ValueError(f"training configuration changed: {directory}")
    env = make_env(config)
    try:
        agent = make_algo("sac", env.obs_dim, env.act_dim, config)
        replay = ReplayBuffer(env.obs_dim, env.act_dim, capacity=1)
        with tempfile.TemporaryDirectory() as temporary:
            logger = Logger(temporary)
            next_iter, steps = load_checkpoint(
                str(directory / "checkpoints"), agent, replay, logger, "sac",
                load_replay_buffer=False,
            )
        if (next_iter, steps, int(agent.update_count)) != (
                MAX_ITERS, TOTAL_STEPS, UPDATES):
            raise ValueError(f"incomplete final checkpoint: {directory}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(
            prefix=f".{destination.name}.", dir=destination.parent))
        try:
            with (temporary / "policy_params.pkl").open("wb") as handle:
                pickle.dump(_to_numpy_tree(nnx.state(agent.policy, nnx.Param)), handle)
            (temporary / "logs").mkdir()
            shutil.copy2(
                directory / "logs/protocol_signature.json",
                temporary / "logs/protocol_signature.json",
            )
            (temporary / "bundle.json").write_text(json.dumps({
                "arm": arm, "seed": seed, "next_iteration": next_iter,
                "total_steps": steps, "update_count": int(agent.update_count),
                "env": config.env_name, "family": config.stochastic_mode_family,
                "backend": config.brax_backend,
            }, indent=2) + "\n")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    finally:
        env.close()
    replay_file = directory / "checkpoints/replay_buffer.npz"
    if replay_file.is_file():
        replay_file.unlink()
    print(f"TRAIN COMPLETE arm={arm} seed={seed}", flush=True)


def load_actor(arm: str, seed: int):
    directory = bundle_dir(arm, seed)
    payload = json.loads((directory / "bundle.json").read_text())
    if (payload["arm"], payload["seed"], payload["total_steps"]) != (
            arm, seed, TOTAL_STEPS):
        raise ValueError(f"invalid policy bundle: {directory}")
    config = final_task_sweep.load_config(directory)
    agent = make_algo("sac", 8, 2, config)
    _patch_flax_variablestate_unpickle()
    with (directory / "policy_params.pkl").open("rb") as handle:
        saved = pickle.load(handle)
    params = _restore_tree_like(
        nnx.state(agent.policy, nnx.Param), saved, "Swimmer actor",
        allow_fallback=False,
    )
    graphdef = nnx.graphdef(agent.policy)

    @jax.jit
    def action(observation):
        policy = nnx.merge(graphdef, params)
        return policy.deterministic(jnp.asarray(observation)[None])[0]

    return lambda observation: np.asarray(action(observation), dtype=np.float32)


def episode(actor, route: str, event_seed: int, index: int) -> dict:
    env = StochasticModeEnv(
        "Swimmer-v2", family="actuator_polarity_2d", backend="generalized",
        dwell_steps=DWELL, seed=event_seed * 10 + index,
        fixed_mode_id=0 if route == "native_reference" else None,
    )
    try:
        if route != "native_reference":
            base = SCHEDULES[event_seed]
            rotation = index % 4
            schedule = base[rotation:] + base[:rotation]
            env.configure_eval_mode_sequence(env.sample_tasks(4), schedule, DWELL)
        observation = env.reset()
        total = 0.0
        used = []
        for step in range(HORIZON):
            mode = env.task_id_for_next_step()
            command = actor(observation)
            if route == "oracle":
                gain = np.asarray(env._tasks[mode]["action_gain"])
                command = gain * command
            observation, reward, done, info = env.step(command)
            if int(info["mode_used"]) != mode:
                raise ValueError("mode used differs from pre-action mode")
            if done:
                raise ValueError(f"Swimmer terminated before horizon at step {step}")
            total += reward
            used.append(mode)
        if route != "native_reference" and len(set(used)) != 4:
            raise ValueError("episode did not visit all four persistent modes")
        return {"return": total, "mode_blocks": [used[k] for k in range(0, HORIZON, DWELL)]}
    finally:
        env.close()


def audit(seed: int) -> None:
    destination = AUDIT_ROOT / f"seed_{seed}" / "audit.json"
    if destination.is_file():
        return
    actors = {arm: load_actor(arm, seed) for arm in ARMS}
    routes = {
        "robust": actors["robust"],
        "uncompensated": actors["reference"],
        "oracle": actors["reference"],
        "native_reference": actors["reference"],
    }
    events = {}
    for event_seed in EVENT_SEEDS:
        events[str(event_seed)] = {}
        for route in ROUTES:
            rows = [episode(routes[route], route, event_seed, index)
                    for index in range(EPISODES)]
            events[str(event_seed)][route] = [row["return"] for row in rows]
        paired = np.asarray(events[str(event_seed)]["oracle"]) - np.asarray(
            events[str(event_seed)]["native_reference"])
        if not np.allclose(paired, 0.0, atol=1e-3):
            raise ValueError(f"oracle/native trajectory mismatch: {event_seed}")
        print(f"AUDIT seed={seed} event={event_seed} complete", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "seed": seed, "event_seeds": list(EVENT_SEEDS),
        "events": events,
        "switching_mean": {
            route: float(np.mean([
                value for event in events.values() for value in event[route]
            ])) for route in ROUTES
        },
        "full_horizon": True, "oracle_native_equivalence": True,
    }, indent=2) + "\n")


def analyze() -> None:
    rows = [json.loads((AUDIT_ROOT / f"seed_{seed}" / "audit.json").read_text())
            for seed in SEEDS]
    differences = []
    seed_passes = []
    for row in rows:
        means = row["switching_mean"]
        robust = means["robust"]
        gain = means["oracle"] - robust
        differences.append(gain)
        seed_passes.append(robust > 0 and gain >= max(5.0, 0.10 * robust))
    robust_mean = float(np.mean([row["switching_mean"]["robust"] for row in rows]))
    oracle_mean = float(np.mean([row["switching_mean"]["oracle"] for row in rows]))
    gain_mean = oracle_mean - robust_mean
    result = {
        "seeds": list(SEEDS), "per_seed": [
            {"seed": row["seed"], **row["switching_mean"],
             "oracle_minus_robust": difference, "headroom_pass": passed}
            for row, difference, passed in zip(rows, differences, seed_passes)
        ],
        "robust_mean": robust_mean, "oracle_mean": oracle_mean,
        "oracle_minus_robust_mean": gain_mean,
        "headroom_pass": bool(
            robust_mean > 0
            and gain_mean >= max(5.0, 0.10 * robust_mean)
            and sum(seed_passes) >= 2
            and all(row["full_horizon"] and row["oracle_native_equivalence"]
                    for row in rows)
        ),
    }
    ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "audit", "analyze"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command in ("train", "audit") and not args.resume:
        parser.error("--resume is required")
    if args.command == "train":
        if args.arm is None or args.seed is None:
            parser.error("train requires --arm and --seed")
        train(args.arm, args.seed)
    elif args.command == "audit":
        if args.seed is None:
            parser.error("audit requires --seed")
        audit(args.seed)
    else:
        analyze()


if __name__ == "__main__":
    main()
