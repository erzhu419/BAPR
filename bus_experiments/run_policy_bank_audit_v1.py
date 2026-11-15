"""Paired stationary and switching headroom audit for one bus policy seed."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

from bus_experiments import policy_bank_headroom_v1 as protocol
from bus_experiments.frozen_policy_bank_core import load_controller, seed_all
from env.sim import env_bus


_CONTROLLERS: dict[str, Any] = {}
_ENVIRONMENTS: dict[str, env_bus] = {}
_ENV_PATH: str = ""


def controller_label(mode: str) -> str:
    return f"specialist:{protocol.require_mode(mode)}"


def _initialize_worker(paths: dict[str, str], env_path: str) -> None:
    global _CONTROLLERS, _ENV_PATH
    torch.set_num_threads(1)
    _ENV_PATH = env_path
    template = env_bus(env_path)
    _CONTROLLERS = {
        label: load_controller(Path(path), template, torch.device("cpu"))
        for label, path in paths.items()
    }


def _environment(kind: str, mode: str | None) -> env_bus:
    key = f"{kind}:{mode or 'switching'}"
    if key not in _ENVIRONMENTS:
        if kind == "stationary":
            _ENVIRONMENTS[key] = env_bus(_ENV_PATH, fixed_mode=mode)
        elif kind == "switching":
            _ENVIRONMENTS[key] = env_bus(
                _ENV_PATH, enable_mode_switch=True,
                mode_switch_interval=(1800, 7200),
                random_initial_mode=True)
        else:
            raise ValueError(f"unknown bus audit environment kind: {kind}")
    return _ENVIRONMENTS[key]


def _evaluate_job(job: dict[str, Any]) -> dict[str, Any]:
    event_seed = int(job["event_seed"])
    seed_all(event_seed)
    kind = str(job["kind"])
    true_mode = job.get("true_mode")
    environment = _environment(kind, true_mode)
    environment.reset()
    state_dict, reward_dict, _ = environment.initialize_state(render=False)
    action_dict = {key: None for key in range(environment.max_agent_num)}
    episode_reward = 0.0
    done = False
    selector = str(job["selector"])
    dynamic_map = dict(job.get("dynamic_map") or {})

    while not done:
        for key in state_dict:
            agent_states = state_dict[key]
            if selector == "dynamic":
                active_label = dynamic_map[environment.current_mode_name]
            else:
                active_label = selector
            policy = _CONTROLLERS[active_label].policy
            if len(agent_states) == 1:
                if action_dict[key] is None:
                    action_dict[key] = policy.get_action(
                        np.asarray(agent_states[0]), deterministic=True)
            elif len(agent_states) == 2:
                if agent_states[0][1] != agent_states[1][1]:
                    episode_reward += float(reward_dict[key])
                state_dict[key] = agent_states[1:]
                action_dict[key] = policy.get_action(
                    np.asarray(state_dict[key][0]), deterministic=True)
        state_dict, reward_dict, done = environment.step(action_dict)

    return {
        "phase": str(job["phase"]),
        "kind": kind,
        "true_mode": true_mode,
        "selector": selector,
        "event_seed": event_seed,
        "reward": float(episode_reward),
        "mode_switch_count": int(environment.mode_switch_count),
        "mode_history": [
            [str(mode), int(timestamp)]
            for mode, timestamp in environment.mode_history
        ],
    }


def _run_jobs(
        jobs: list[dict[str, Any]], paths: dict[str, str], workers: int,
        env_path: str) -> list[dict[str, Any]]:
    context = mp.get_context("spawn")
    with context.Pool(
            processes=max(1, int(workers)),
            initializer=_initialize_worker,
            initargs=(paths, env_path)) as pool:
        return list(pool.map(_evaluate_job, jobs, chunksize=1))


def _mean(rows: list[dict[str, Any]]) -> float:
    if not rows:
        raise ValueError("cannot aggregate an empty bus audit cell")
    return float(np.mean([float(row["reward"]) for row in rows]))


def _relative_gain(candidate: float, baseline: float) -> float:
    return float((candidate - baseline) / max(abs(baseline), 1.0))


def run_audit(seed: int, workers: int) -> dict[str, Any]:
    protocol.validate_registration()
    seed = protocol.require_seed(seed)
    paths = {"robust": str(protocol.source_controller(seed))}
    paths.update({
        controller_label(mode): str(protocol.specialist_controller(seed, mode))
        for mode in protocol.MODES
    })
    missing = [path for path in paths.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing bus controller bundles: {missing}")
    env_path = str(protocol.ROOT / "env")
    labels = tuple(paths)

    calibration_jobs = [
        {
            "phase": "calibration",
            "kind": "stationary",
            "true_mode": mode,
            "selector": label,
            "event_seed": event_seed,
        }
        for mode in protocol.MODES
        for label in labels
        for event_seed in protocol.CALIBRATION_EVENT_SEEDS
    ]
    calibration_rows = _run_jobs(
        calibration_jobs, paths, workers, env_path)
    calibration_cells: dict[str, dict[str, float]] = {}
    selected: dict[str, str] = {}
    for mode in protocol.MODES:
        calibration_cells[mode] = {}
        for label in labels:
            rows = [
                row for row in calibration_rows
                if row["true_mode"] == mode and row["selector"] == label
            ]
            calibration_cells[mode][label] = _mean(rows)
        selected[mode] = max(
            labels, key=lambda label: calibration_cells[mode][label])

    holdout_jobs = [
        {
            "phase": "stationary_holdout",
            "kind": "stationary",
            "true_mode": mode,
            "selector": label,
            "event_seed": event_seed,
        }
        for mode in protocol.MODES
        for label in labels
        for event_seed in protocol.STATIONARY_HOLDOUT_EVENT_SEEDS
    ]
    switching_selectors = ("robust", "dynamic", *(
        controller_label(mode) for mode in protocol.MODES))
    switching_jobs = [
        {
            "phase": "switching_holdout",
            "kind": "switching",
            "true_mode": None,
            "selector": selector,
            "dynamic_map": selected if selector == "dynamic" else None,
            "event_seed": event_seed,
        }
        for selector in switching_selectors
        for event_seed in protocol.SWITCHING_EVENT_SEEDS
    ]
    holdout_rows = _run_jobs(
        holdout_jobs + switching_jobs, paths, workers, env_path)
    stationary_rows = [
        row for row in holdout_rows if row["kind"] == "stationary"]
    switching_rows = [
        row for row in holdout_rows if row["kind"] == "switching"]

    stationary: dict[str, Any] = {}
    stationary_mode_wins = 0
    robust_stationary_rewards = []
    oracle_stationary_rewards = []
    for mode in protocol.MODES:
        matrix = {}
        for label in labels:
            rows = [
                row for row in stationary_rows
                if row["true_mode"] == mode and row["selector"] == label
            ]
            matrix[label] = {
                "reward_mean": _mean(rows),
                "rewards": [float(row["reward"]) for row in rows],
            }
        robust_mean = float(matrix["robust"]["reward_mean"])
        oracle_label = selected[mode]
        oracle_mean = float(matrix[oracle_label]["reward_mean"])
        gain = _relative_gain(oracle_mean, robust_mean)
        mode_pass = gain >= protocol.MIN_STATIONARY_MODE_GAIN
        stationary_mode_wins += int(mode_pass)
        robust_stationary_rewards.extend(matrix["robust"]["rewards"])
        oracle_stationary_rewards.extend(matrix[oracle_label]["rewards"])
        stationary[mode] = {
            "selected_controller": oracle_label,
            "robust_reward_mean": robust_mean,
            "oracle_reward_mean": oracle_mean,
            "relative_gain": gain,
            "passes_gain": mode_pass,
            "controller_matrix": matrix,
        }

    switching: dict[str, Any] = {}
    for selector in switching_selectors:
        rows = [row for row in switching_rows if row["selector"] == selector]
        switching[selector] = {
            "reward_mean": _mean(rows),
            "rewards": [float(row["reward"]) for row in rows],
            "rows": rows,
        }
    robust_switch = float(switching["robust"]["reward_mean"])
    dynamic_switch = float(switching["dynamic"]["reward_mean"])
    switching_gain = _relative_gain(dynamic_switch, robust_switch)
    robust_by_seed = {
        int(row["event_seed"]): float(row["reward"])
        for row in switching["robust"]["rows"]
    }
    dynamic_by_seed = {
        int(row["event_seed"]): float(row["reward"])
        for row in switching["dynamic"]["rows"]
    }
    switching_event_wins = sum(
        dynamic_by_seed[event_seed] > robust_by_seed[event_seed]
        for event_seed in protocol.SWITCHING_EVENT_SEEDS
    )
    best_fixed_label = max(
        (label for label in switching if label != "dynamic"),
        key=lambda label: switching[label]["reward_mean"])
    best_fixed_mean = float(switching[best_fixed_label]["reward_mean"])
    dynamic_beats_every_fixed = dynamic_switch > best_fixed_mean

    stationary_robust_mean = float(np.mean(robust_stationary_rewards))
    stationary_oracle_mean = float(np.mean(oracle_stationary_rewards))
    passed = bool(
        stationary_mode_wins >= protocol.MIN_STATIONARY_MODE_WINS
        and switching_gain >= protocol.MIN_SWITCHING_GAIN
        and switching_event_wins == protocol.REQUIRED_SWITCHING_EVENT_WINS
        and dynamic_beats_every_fixed
    )
    return {
        "schema": protocol.AUDIT_SCHEMA,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "training_seed": seed,
        "policy_selection": selected,
        "calibration": calibration_cells,
        "stationary_holdout": stationary,
        "stationary_summary": {
            "robust_reward_mean": stationary_robust_mean,
            "oracle_reward_mean": stationary_oracle_mean,
            "relative_gain": _relative_gain(
                stationary_oracle_mean, stationary_robust_mean),
            "mode_wins_at_threshold": stationary_mode_wins,
            "required_mode_wins": protocol.MIN_STATIONARY_MODE_WINS,
        },
        "switching_holdout": switching,
        "switching_summary": {
            "robust_reward_mean": robust_switch,
            "dynamic_oracle_reward_mean": dynamic_switch,
            "relative_gain": switching_gain,
            "event_wins": switching_event_wins,
            "required_event_wins": protocol.REQUIRED_SWITCHING_EVENT_WINS,
            "best_fixed_controller": best_fixed_label,
            "best_fixed_reward_mean": best_fixed_mean,
            "dynamic_beats_every_fixed": dynamic_beats_every_fixed,
        },
        "passes_policy_seed_gate": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    result = run_audit(args.seed, args.workers)
    destination = protocol.audit_dir(args.seed)
    protocol.write_json_atomic(destination / "audit.json", result)
    protocol.write_json_atomic(destination / "audit_manifest.json", {
        "schema": "bapr.bus-policy-bank-audit-manifest.v1",
        "training_seed": int(args.seed),
        "complete": True,
        "passed": bool(result["passes_policy_seed_gate"]),
        "result": protocol.file_record(destination / "audit.json"),
    })
    print(
        "BUS_POLICY_BANK_AUDIT_COMPLETE "
        f"seed={args.seed} pass={result['passes_policy_seed_gate']} "
        f"stationary_gain={result['stationary_summary']['relative_gain']:.3f} "
        f"switching_gain={result['switching_summary']['relative_gain']:.3f}",
        flush=True)


if __name__ == "__main__":
    main()
