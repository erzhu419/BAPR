#!/usr/bin/env python3
"""Audit that BAPR logical env params change Brax physics fields.

This catches the easy-to-miss Brax no-op where replacing legacy top-level
fields such as ``body_mass`` or ``dof_damping`` leaves the pipeline physics
unchanged. The script intentionally checks the actual fields read by Brax:
``gravity``, ``link.inertia.mass``, ``dof.damping``, and ``link.inertia.i``.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any


os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np

from jax_experiments.envs.brax_env import (  # noqa: E402
    BraxNonstationaryEnv,
    RAND_PARAMS_MAP,
)
from jax_experiments.envs.discrete_mode_env import (  # noqa: E402
    DiscreteModePiecewiseEnv,
)


DEFAULT_ENVS = ["HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Ant-v2"]
LOGICAL_PARAMS = ["gravity", "body_mass", "dof_damping", "body_inertia"]


@dataclass
class CheckResult:
    env_name: str
    protocol: str
    logical_param: str
    physical_path: str
    case: str
    max_abs_diff: float
    ok: bool


def get_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def max_abs_diff(a: Any, b: Any) -> float:
    return float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b))))


def continuous_checks(env_name: str, multiplier: float) -> list[CheckResult]:
    results: list[CheckResult] = []
    env = BraxNonstationaryEnv(
        env_name,
        rand_params=list(LOGICAL_PARAMS),
        log_scale_limit=0.0,
        seed=0,
        backend="spring",
    )
    for param in LOGICAL_PARAMS:
        physical_path = RAND_PARAMS_MAP[param]
        base = np.asarray(env._base_values[param])
        env.set_task({param: base * multiplier})
        diff = max_abs_diff(
            get_path(env.base_sys, physical_path),
            get_path(env._current_sys, physical_path),
        )
        results.append(
            CheckResult(
                env_name=env_name,
                protocol="continuous",
                logical_param=param,
                physical_path=physical_path,
                case=f"x{multiplier:g}",
                max_abs_diff=diff,
                ok=diff > 1e-6,
            )
        )
    return results


def discrete_checks(env_name: str) -> list[CheckResult]:
    results: list[CheckResult] = []
    env = DiscreteModePiecewiseEnv(
        env_name,
        mean_dwell_iters=60,
        steps_per_iter=4000,
        seed=0,
        backend="spring",
    )
    for mode_key, mode_dict, mode_sys in zip(env.mode_keys, env.modes, env._mode_sys):
        for param in LOGICAL_PARAMS:
            if param not in mode_dict:
                continue
            physical_path = RAND_PARAMS_MAP[param]
            diff = max_abs_diff(
                get_path(env.base_sys, physical_path),
                get_path(mode_sys, physical_path),
            )
            results.append(
                CheckResult(
                    env_name=env_name,
                    protocol="discrete_mode",
                    logical_param=param,
                    physical_path=physical_path,
                    case=mode_key,
                    max_abs_diff=diff,
                    ok=diff > 1e-6,
                )
            )
    return results


def print_results(results: list[CheckResult]) -> None:
    header = (
        f"{'env':<16} {'protocol':<14} {'case':<16} {'logical':<13} "
        f"{'physical':<20} {'max_abs_diff':>13} status"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(
            f"{r.env_name:<16} {r.protocol:<14} {r.case:<16} "
            f"{r.logical_param:<13} {r.physical_path:<20} "
            f"{r.max_abs_diff:13.6g} {status}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", action="append", dest="envs", default=None,
                        help="Env to check; repeatable. Default: all main envs.")
    parser.add_argument("--multiplier", type=float, default=1.2,
                        help="Continuous-task multiplier for field-change checks.")
    parser.add_argument("--continuous-only", action="store_true")
    parser.add_argument("--discrete-only", action="store_true")
    args = parser.parse_args()

    if args.continuous_only and args.discrete_only:
        parser.error("--continuous-only and --discrete-only are mutually exclusive")

    all_results: list[CheckResult] = []
    for env_name in args.envs or DEFAULT_ENVS:
        if not args.discrete_only:
            all_results.extend(continuous_checks(env_name, args.multiplier))
        if not args.continuous_only:
            all_results.extend(discrete_checks(env_name))

    print_results(all_results)
    failures = [r for r in all_results if not r.ok]
    if failures:
        print(f"\nFAIL: {len(failures)} physics-field checks did not change.", file=sys.stderr)
        return 1
    print(f"\nPASS: {len(all_results)} physics-field checks changed real Brax fields.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
