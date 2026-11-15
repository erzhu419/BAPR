"""Run one frozen training arm or strict CPU audit; never retune old results."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from scipy.stats import t

from jax_experiments.analysis import regularized_comparison_protocol_v39 as p
from jax_experiments.analysis import final_task_sweep
from jax_experiments.analysis import run_regime_polarity_action_compensation_source_v32 as source_template
from jax_experiments.analysis import run_regime_polarity_action_compensation_reference_v32 as reference_template
from jax_experiments.analysis import run_regime_polarity_action_compensation_baseline_v32 as baseline_template
from jax_experiments.analysis import run_regime_polarity_specialist_policy_stability_v20 as fork
from jax_experiments.analysis import run_regime_polarity_v5_final_comparison_audit_v18 as evaluate
from jax_experiments.analysis import run_regime_polarity_action_compensation_audit_v28 as compensation
from jax_experiments.analysis import regime_polarity_specialist_expected_action_model_v5 as inverse
from jax_experiments.algos.sac_policy_stability import SACPolicyStability
from jax_experiments.common.checkpoint import (
    _patch_flax_variablestate_unpickle, _restore_tree_like, _to_numpy_tree,
    load_checkpoint, save_checkpoint,
)
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.train import make_algo, make_env


def read(path):
    return json.loads(path.read_text())


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(stream.name, path)


def set_option(command, name, value):
    flag = "--" + name
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend((flag, str(value)))


def training_command(arm, seed):
    directory = p.run_dir(arm, seed)
    template_seed = p.old.TRAINING_SEEDS[0]
    if arm == "source":
        command = source_template.training_command(template_seed)
    elif arm == "reference":
        command = reference_template.training_command(p.old.REFERENCE_VARIANT, template_seed, 0, directory)
    else:
        kind = {"sac": "sac_replica", "escp": "escp_recurrent",
                "resac_zero": "resac_b0", "resac_full": "resac_b0"}[arm]
        command = baseline_template.training_command(kind, template_seed,
                    p.old.LONG_SAC_SLOT if arm == "sac" else None)
    command[0] = sys.executable
    for name, value in {"seed": seed, "save_root": directory.parent,
                        "run_name": directory.name}.items():
        set_option(command, name, value)
    if arm == "resac_full":
        for flag in ("--resac_adaptive_beta", "--use_ema_eval"):
            command.remove(flag)
        for name, value in {"weight_reg": 0.01, "beta_ood": 0.01,
                            "beta": -2.0, "resac_beta_bc": 0.001,
                            "resac_independent_ratio": 1.0, "resac_anchor_lambda": 0.0}.items():
            set_option(command, name, value)
    return command


def check_budget(metadata, arm, seed):
    if metadata["identity"] != p.identity(arm, seed) or metadata["runtime_versions"] != p.RUNTIME:
        raise ValueError(f"different identity/runtime in {arm}/{seed}")
    if metadata["budget"] != p.budget(arm):
        raise ValueError(f"incomplete budget in {arm}/{seed}: {metadata['budget']}")
    if arm == "resac_full":
        settings = {"weight_reg": 0.01, "beta_ood": 0.01, "beta": -2.0,
                    "resac_beta_bc": 0.001, "resac_independent_ratio": 1.0,
                    "resac_adaptive_beta": False, "resac_anchor_lambda": 0.0,
                    "use_ema_eval": False}
        if any(metadata["config"].get(key) != value for key, value in settings.items()):
            raise ValueError("regularized RE-SAC settings differ")
    if arm == "resac_zero" and any(metadata["config"][key] != 0 for key in ("weight_reg", "beta_ood")):
        raise ValueError("zero-regularization comparator settings differ")


def build_agent(config, obs_dim, act_dim, arm):
    if arm == "reference":
        return SACPolicyStability(obs_dim, act_dim, config, seed=config.seed)
    return make_algo(config.algo, obs_dim, act_dim, config)


def publish(arm, seed):
    run = p.run_dir(arm, seed)
    signature = read(run / "logs/protocol_signature.json")
    config = final_task_sweep.load_config(run)
    env = make_env(config, seed_offset=0)
    try:
        agent = build_agent(config, env.obs_dim, env.act_dim, arm)
        if hasattr(agent, "set_task_metadata"):
            agent.set_task_metadata(env.sample_tasks(4))
        replay = ReplayBuffer(env.obs_dim, env.act_dim, capacity=1)
        with tempfile.TemporaryDirectory() as temporary:
            iteration, steps = load_checkpoint(str(run / "checkpoints"), agent, replay,
                Logger(temporary), config.algo, load_replay_buffer=False)
        metadata = {"identity": p.identity(arm, seed), "runtime_versions": p.check_runtime(),
                    "budget": {"next_iteration": iteration, "total_steps": steps,
                               "update_count": int(agent.update_count)}, "config": signature["config"],
                    "python": sys.executable, "host": signature["host"]}
        check_budget(metadata, arm, seed)
        states = [nnx.state(getattr(agent, name), nnx.Param) for name in ("policy", "critic", "target_critic")]
        if hasattr(agent, "context_net"):
            states.append(nnx.state(agent.context_net, nnx.Param))
        if not all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves([*states, agent.log_alpha])):
            raise RuntimeError(f"PROTOCOL_INTEGRITY: nonfinite trained parameters in {arm}/{seed}")
        payload = {"policy": _to_numpy_tree(nnx.state(agent.policy, nnx.Param))}
        if arm == "source":
            payload.update(critic=_to_numpy_tree(nnx.state(agent.critic, nnx.Param)),
                           target_critic=_to_numpy_tree(nnx.state(agent.target_critic, nnx.Param)),
                           log_alpha=_to_numpy_tree(agent.log_alpha))
        if arm == "escp":
            payload.update(context_net=_to_numpy_tree(nnx.state(agent.context_net, nnx.Param)),
                           custom_agent_state=agent.checkpoint_state())
        if arm == "resac_zero":
            payload["ema_policy"] = _to_numpy_tree(nnx.state(agent.ema_policy, nnx.Param))
        destination = p.bundle_dir(arm, seed)
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "parameters.pkl").open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        logs = destination / "logs"
        logs.mkdir(exist_ok=True)
        shutil.copy2(run / "logs/protocol_signature.json", logs / "protocol_signature.json")
        write(destination / "complete.json", metadata)
    finally:
        if hasattr(env, "close"):
            env.close()


def restore(module, state, name):
    nnx.update(module, _restore_tree_like(nnx.state(module, nnx.Param), state, name, allow_fallback=False))


def load_agent(arm, seed):
    directory = p.bundle_dir(arm, seed)
    metadata = read(directory / "complete.json")
    check_budget(metadata, arm, seed)
    config = final_task_sweep.load_config(directory)
    env = make_env(config, seed_offset=0)
    try:
        agent = build_agent(config, env.obs_dim, env.act_dim, arm)
        if hasattr(agent, "set_task_metadata"):
            agent.set_task_metadata(env.sample_tasks(4))
        with (directory / "parameters.pkl").open("rb") as stream:
            payload = pickle.load(stream)
        for name in ("policy", "critic", "target_critic", "context_net", "ema_policy"):
            if name in payload:
                restore(getattr(agent, name), payload[name], f"{arm} {name}")
        if "log_alpha" in payload:
            agent.log_alpha = jnp.asarray(payload["log_alpha"])
        if "custom_agent_state" in payload:
            agent.load_checkpoint_state(payload["custom_agent_state"])
        agent.update_count = metadata["budget"]["update_count"]
        return config, agent
    finally:
        if hasattr(env, "close"):
            env.close()


def bootstrap_reference(seed):
    run = p.run_dir("reference", seed)
    marker = run / "checkpoints/bootstrap.json"
    if marker.is_file():
        if read(marker)["identity"] != p.identity("reference", seed):
            raise ValueError("reference bootstrap identity differs")
        return
    source_config, source = load_agent("source", seed)
    config = copy.deepcopy(source_config)
    config.max_iters, config.start_train_steps = 2100, 0
    config.save_root, config.run_name = str(run.parent), run.name
    config.stochastic_mode_fixed_id = 0
    config.sac_actor_update_period, config.sac_select_best_eval = 1, False
    config.resume, config.min_resume_iteration = True, 1400
    agent = SACPolicyStability(source.obs_dim, source.act_dim, config, seed=seed)
    for name in ("policy", "critic", "target_critic"):
        nnx.update(getattr(agent, name), nnx.state(getattr(source, name), nnx.Param))
    agent.log_alpha = jnp.asarray(source.log_alpha)
    fork.historical._reset_optimizers(agent)
    agent.update_count = 350000
    agent.reset_selection_anchor()
    equivalence = fork._controller_equivalence(source, agent)
    if not equivalence["pass"]:
        raise ValueError(f"reference fork differs: {equivalence}")
    save_checkpoint(str(run / "checkpoints"), agent,
                    ReplayBuffer(source.obs_dim, source.act_dim, capacity=config.replay_size),
                    Logger(str(run / "logs")), 1399, 5600000, "sac")
    write(marker, {"identity": p.identity("reference", seed), "source_budget": p.budget("source"),
                   "controller_equivalence": equivalence, "replay_and_optimizers_reset": True})


def train(arm, seed):
    destination = p.bundle_dir(arm, seed) / "complete.json"
    if destination.is_file():
        check_budget(read(destination), arm, seed)
        return
    if arm == "reference":
        bootstrap_reference(seed)
    environment = dict(os.environ, PYTHONPATH=str(p.ROOT), BAPR_SAC_ACTOR_UPDATE_PERIOD="1",
                       BAPR_SAC_SELECT_BEST_EVAL="0")
    command = training_command(arm, seed)
    print("TRAIN", " ".join(command), flush=True)
    subprocess.run(command, cwd=p.ROOT, env=environment, check=True)
    publish(arm, seed)
    print(f"COMPLETE {arm}/{seed}: {p.budget(arm)}", flush=True)


def install_evaluation():
    evaluate.protocol = p
    compensation.protocol, compensation.base, compensation.v5_model = p, evaluate, inverse
    inverse.protocol.MODEL_PATH = p.ESTIMATOR / "inverse_params.npz"
    inverse.protocol.MODEL_MANIFEST = p.ESTIMATOR / "model_manifest.json"


def audit(seed):
    install_evaluation()
    controllers = {arm: load_agent(arm, seed) for arm in p.TRAIN_ARMS}
    reference_config, reference = controllers["reference"]
    action = jax.jit(lambda obs: reference.policy.deterministic(obs[None])[0])
    stack = {"config": reference_config, "reference_agent": reference,
             "actions": {"specialist_0": lambda obs: np.asarray(action(jnp.asarray(obs)))}}
    runtimes = {arm: (config, evaluate.corrected_audit.AgentRuntime(agent, config))
                for arm, (config, agent) in controllers.items() if arm != "reference"}
    stationary, switching = {}, {}
    for event in p.STATIONARY_HOLDOUT_EVENT_SEEDS:
        rows = {arm: {} for arm in p.EVAL_ARMS}
        for mode in p.MODES:
            for arm, (config, runtime) in runtimes.items():
                rows[arm][str(mode)] = evaluate._stationary_runtime(config, runtime, event, mode)
            rows["no_compensation"][str(mode)] = evaluate._stationary_direct(
                reference_config, stack["actions"]["specialist_0"], event, mode, p.STATIONARY_EPISODES_PER_MODE)
            for arm, route in (("oracle", "true_mode"), ("bapr", "v5_posterior")):
                rows[arm][str(mode)] = compensation._compensated_stationary(stack, 0, route, event, mode)
        stationary[str(event)] = rows
        print(f"STATIONARY event={event} complete", flush=True)
    for event in p.SWITCHING_EVENT_SEEDS:
        rows = {arm: evaluate._switching_runtime(config, runtime, event)
                for arm, (config, runtime) in runtimes.items()}
        rows["no_compensation"] = evaluate._switching_runtime(reference_config,
            evaluate.corrected_audit.AgentRuntime(reference, reference_config), event)
        for arm, route in (("oracle", "true_mode"), ("bapr", "v5_posterior")):
            rows[arm] = compensation._compensated_switching(stack, 0, route, event)
        if len({row["mode_trace_sha256"] for row in rows.values()}) != 1:
            raise ValueError("comparison arms saw different switching schedules")
        switching[str(event)] = rows
        print(f"SWITCHING event={event} complete", flush=True)
    write(p.audit_path(seed), {"protocol": p.PROTOCOL, "seed": seed, "runtime_versions": p.check_runtime(),
          "budgets": {arm: read(p.bundle_dir(arm, seed) / "complete.json")["budget"] for arm in p.TRAIN_ARMS},
          "stationary": stationary, "switching": switching})


def analyze():
    audits = [read(p.audit_path(seed)) for seed in p.SEEDS]
    values = {arm: np.asarray([np.mean([event[arm]["returns"] for event in row["switching"].values()])
                              for row in audits]) for arm in p.EVAL_ARMS}
    for seed, row in zip(p.SEEDS, audits, strict=True):
        if row["seed"] != seed or row["protocol"] != p.PROTOCOL or row["runtime_versions"] != p.RUNTIME:
            raise ValueError("audit identity/runtime differs")
        if row["budgets"] != {arm: p.budget(arm) for arm in p.TRAIN_ARMS}:
            raise ValueError("audit budgets differ")
    comparisons = {}
    for arm in ("resac_full", "sac", "escp", "resac_zero"):
        delta = values["bapr"] - values[arm]
        radius = float(t.ppf(0.975, 4) * delta.std(ddof=1) / np.sqrt(5))
        comparisons[arm] = {"paired_mean": float(delta.mean()),
                            "ci95": [float(delta.mean()-radius), float(delta.mean()+radius)],
                            "seed_wins": int(np.sum(delta > 0)), "primary": arm == "resac_full"}
    write(p.ANALYSIS, {"protocol": p.PROTOCOL, "policy_seeds": list(p.SEEDS),
          "runtime_versions": p.RUNTIME, "switching_means": {arm: float(row.mean()) for arm, row in values.items()},
          "comparisons": comparisons, "historical_cohorts_pooled": False,
          "estimator_refitted": False, "total_data_matched": False})
    print(json.dumps(comparisons, indent=2), flush=True)


def smoke():
    seed = p.SEEDS[0]
    rows = {}
    for arm in ("sac", "escp", "resac_zero", "resac_full"):
        command = training_command(arm, seed)
        directory = p.RUN_ROOT / "smoke" / arm
        for name, value in {"max_iters": 2, "samples_per_iter": 4000, "updates_per_iter": 250,
                            "start_train_steps": 0, "initial_random_steps": 0,
                            "save_root": directory.parent, "run_name": directory.name,
                            "log_interval": 100, "eval_episodes": 1}.items():
            set_option(command, name, value)
        subprocess.run(command, cwd=p.ROOT, check=True)
        with (directory / "checkpoints/train_state.pkl").open("rb") as stream:
            checkpoint = pickle.load(stream)
        if (checkpoint["iteration"], checkpoint["total_steps"]) != (1, 8000):
            raise ValueError("smoke did not finish its rollout")
        with (directory / "checkpoints/params.pkl").open("rb") as stream:
            parameters = pickle.load(stream)
        if parameters["update_count"] != 500:
            raise ValueError("smoke did not finish its updates")
        metrics = {name: np.load(directory / "logs" / f"{name}.npy").tolist()
                   for name in ("critic_loss", "policy_loss")}
        if not all(np.isfinite(value).all() for value in metrics.values()):
            raise ValueError("nonfinite smoke updates")
        if arm == "resac_full":
            bonus = np.load(directory / "logs/reg_bonus_mean.npy")
            if not np.isfinite(bonus).all() or not (bonus > 0).all():
                raise ValueError("RE-SAC smoke did not retain the positive regularization shift")
            metrics["reg_bonus_mean"] = bonus.tolist()
        rows[arm] = {"config": read(directory / "logs/protocol_signature.json")["config"], "metrics": metrics}
    write(p.SMOKE, {"runtime_versions": p.check_runtime(), "arms": rows,
                    "memory_stats": jax.devices()[0].memory_stats()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train", "audit", "analyze"))
    parser.add_argument("--arm", choices=p.TRAIN_ARMS)
    parser.add_argument("--seed", type=int, choices=p.SEEDS)
    args = parser.parse_args()
    p.check_runtime()
    _patch_flax_variablestate_unpickle()
    if args.mode in ("smoke", "train") and jax.default_backend() != "gpu":
        raise ValueError("training requires the admitted GPU runtime")
    if args.mode == "smoke":
        smoke()
    elif args.mode == "train":
        train(args.arm, args.seed)
    elif args.mode == "audit":
        audit(args.seed)
    else:
        analyze()


if __name__ == "__main__":
    main()
