"""Original Torch RE-SAC trainer on the frozen Brax actuator benchmark."""
from copy import deepcopy
from functools import partial
from importlib.metadata import version
import argparse
import json
import random
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis import submission_extensions_protocol_v47 as p
from jax_experiments.analysis import regularized_comparison_protocol_v39 as historical
from jax_experiments.analysis import run_mechanism_gap_v34 as common
from jax_experiments.analysis.run_submission_extensions_v47 import check_runtime, stationary
from jax_experiments.networks.policy import GaussianPolicy

read = lambda path: json.loads(path.read_text())


def directory(arm, seed):
    return p.BUNDLE_ROOT / "source" / arm / f"seed_{seed}"


def source_imports():
    sys.path.insert(0, str(p.SOURCE))
    import gym
    import torch
    import yaml
    import rlkit.torch.pytorch_util as ptu
    from rlkit.torch.sac.policies import TanhGaussianPolicy
    from rlkit.data_management.torch_replay_buffer import TorchReplayBuffer
    from resac_networks import VectorizedQNetwork
    from resac_trainer import RESACTrainer
    if torch.__version__ != "2.4.1+cu121" or version("gym") != "0.10.9":
        raise ValueError("source trainer runtime differs from the native Torch calibration")
    config = yaml.safe_load((p.SOURCE / "halfcheetah.yaml").read_text())
    return locals()


def actor_from_torch(policy, obs_dim=17, act_dim=6):
    model = GaussianPolicy(obs_dim, act_dim, 256, rngs=nnx.Rngs(0))
    for target, source in zip([*model.layers, model.mean_head, model.log_std_head],
                              [*policy.fcs, policy.last_fc, policy.last_fc_log_std], strict=True):
        target.kernel.value = jnp.asarray(source.weight.detach().cpu().numpy().T)
        target.bias.value = jnp.asarray(source.bias.detach().cpu().numpy())
    return nnx.graphdef(model), nnx.state(model, nnx.Param)


def build_collector(world, graph):
    @partial(jax.jit, static_argnums=(3, 4))
    def collect(params, key, modes, count, warmup):
        reset_key, rollout_key = jax.random.split(key)

        def body(carry, pair):
            state, position = carry
            mode, key = pair
            action_key, noise_key, reset_key = jax.random.split(key, 3)
            if warmup:
                action = jax.random.uniform(action_key, (world.act_dim,), minval=-1, maxval=1)
            else:
                action = nnx.merge(graph, params).sample(state.obs[None], action_key)[0][0]
            executed = common.apply_action_disturbance(action, noise_key, world.gains[mode], .02)
            following = world.step(state, executed)
            restart = following.done > .5
            outgoing = jax.lax.cond(restart, lambda: world.reset(world.sys, reset_key), lambda: following)
            return (outgoing, jnp.where(restart, 0, position + 1)), (
                state.obs, action, following.reward, following.obs, following.done)

        return jax.lax.scan(body, (world.reset(world.sys, reset_key), jnp.asarray(0)),
                            (modes, jax.random.split(rollout_key, count)))[1]
    return collect


class SourceRun:
    def __init__(self, arm, seed, api, capacity=1000000):
        self.arm, self.seed, self.api = arm, seed, api
        torch, gym, ptu = [api[name] for name in ("torch", "gym", "ptu")]
        torch.set_num_threads(2)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        ptu.set_gpu_mode(True, 0)
        self.config = deepcopy(api["config"])
        self.config["trainer_kwargs"]["weight_reg"] = .01 if arm == "full" else 0.
        self.world = common.World("hc", seed, physics_unroll=1)
        env = type("Dimensions", (), {"observation_space": gym.spaces.Box(-np.ones(17), np.ones(17)),
                                       "action_space": gym.spaces.Box(-np.ones(6), np.ones(6))})()
        policy = api["TanhGaussianPolicy"]([256, 256], 17, 6)
        critic = api["VectorizedQNetwork"](17, 6, 256, 10)
        self.trainer = api["RESACTrainer"](env, policy, critic, deepcopy(critic),
            ensemble_size=10, **self.config["trainer_kwargs"])
        for network in self.trainer.networks:
            network.to(ptu.device)
        self.replay = api["TorchReplayBuffer"](capacity, env)
        self.graph, unused = actor_from_torch(policy)
        self.collect = build_collector(self.world, self.graph)
        self.next_iteration = self.steps = 0
        self.modes = common.training_modes(seed, p.source_budget()["total_steps"], 250)
        self.optimizer_counts = {"critic": 0, "actor": 0}
        for name, optimizer in (("critic", self.trainer.qf_optimizer), ("actor", self.trainer.policy_optimizer)):
            optimizer.register_step_post_hook(lambda opt, args, kwargs, key=name: self.count_step(key))

    def count_step(self, key):
        self.optimizer_counts[key] += 1

    def rollout(self, count, iteration, warmup=False):
        torch = self.api["torch"]
        graph, params = actor_from_torch(self.trainer.policy)
        modes = self.modes[self.steps:self.steps+count]
        key = jax.random.fold_in(jax.random.PRNGKey(self.seed), iteration + 1)
        values = self.collect(params, key, jnp.asarray(modes), count, warmup)
        fields = ("_observations", "_actions", "_rewards", "_next_obs", "_terminals")
        indexes = (torch.arange(count) + self.replay._top) % self.replay._max_replay_buffer_size
        for field, value in zip(fields, values, strict=True):
            array = np.asarray(value).reshape(count, -1).copy()
            if not np.isfinite(array).all():
                raise ValueError("nonfinite source training transitions")
            getattr(self.replay, field)[indexes] = torch.from_numpy(array)
        self.replay._top = int((self.replay._top + count) % self.replay._max_replay_buffer_size)
        self.replay._size = min(self.replay._size + count, self.replay._max_replay_buffer_size)
        self.steps += count

    def update(self, count=250):
        for unused in range(count):
            self.trainer.train(self.replay.next_batch(256))
        self.api["torch"].cuda.synchronize()
        metrics = {key: float(value) for key, value in self.trainer.get_diagnostics().items()}
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError(f"nonfinite original trainer metrics: {metrics}")
        self.trainer.end_epoch(self.next_iteration)
        return metrics

    def state(self):
        from jax_experiments.analysis.run_resac_torch_native_v43 import cpu_tree
        torch, trainer = self.api["torch"], self.trainer
        return {"protocol": p.PROTOCOL, "arm": self.arm, "seed": self.seed, "config": self.config,
            "runtime_versions": check_runtime(), "torch": torch.__version__,
            "next_iteration": self.next_iteration, "total_steps": self.steps,
            "models": [cpu_tree(model.state_dict(), torch) for model in trainer.networks],
            "optimizers": [cpu_tree(opt.state_dict(), torch) for opt in
                (trainer.qf_optimizer, trainer.policy_optimizer, trainer.alpha_optimizer)],
            "log_alpha": cpu_tree(trainer.log_alpha, torch), "scaler": trainer.scaler.state_dict(),
            "critic_calls": trainer._n_train_steps_total, "trainer_calls": trainer._num_train_steps,
            "optimizer_counts": dict(self.optimizer_counts),
            "replay": {key: cpu_tree(getattr(self.replay, key), torch) for key in
                ("_observations", "_next_obs", "_actions", "_rewards", "_terminals")},
            "replay_top": self.replay._top, "replay_size": self.replay._size,
            "prefetch": cpu_tree(self.replay.batch, torch),
            "rng": {"numpy": np.random.get_state(), "python": random.getstate(),
                    "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}}

    def restore(self, saved):
        torch, ptu, trainer = self.api["torch"], self.api["ptu"], self.trainer
        if (saved["protocol"], saved["arm"], saved["seed"], saved["config"], saved["runtime_versions"], saved["torch"]) != (
                p.PROTOCOL, self.arm, self.seed, self.config, p.RUNTIME, torch.__version__):
            raise ValueError("source checkpoint identity/runtime differs")
        for model, state in zip(trainer.networks, saved["models"], strict=True):
            model.load_state_dict(state)
        for optimizer, state in zip((trainer.qf_optimizer, trainer.policy_optimizer, trainer.alpha_optimizer), saved["optimizers"], strict=True):
            optimizer.load_state_dict(state)
        trainer.log_alpha.data.copy_(saved["log_alpha"].to(ptu.device))
        trainer.scaler.load_state_dict(saved["scaler"])
        trainer._n_train_steps_total, trainer._num_train_steps = saved["critic_calls"], saved["trainer_calls"]
        trainer._cached_reg_norm = None
        for key, tensor in saved["replay"].items():
            getattr(self.replay, key).copy_(tensor)
        self.replay._top, self.replay._size = saved["replay_top"], saved["replay_size"]
        self.replay.batch = {key: tensor.to(ptu.device) for key, tensor in saved["prefetch"].items()}
        self.optimizer_counts = dict(saved["optimizer_counts"])
        self.next_iteration, self.steps = saved["next_iteration"], saved["total_steps"]
        np.random.set_state(saved["rng"]["numpy"])
        random.setstate(saved["rng"]["python"])
        torch.set_rng_state(saved["rng"]["torch"])
        torch.cuda.set_rng_state(saved["rng"]["cuda"])
        trainer.end_epoch(self.next_iteration)
        print(f"RESUME iter={self.next_iteration} steps={self.steps} updates={trainer._n_train_steps_total}", flush=True)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        self.api["torch"].save(self.state(), temporary)
        temporary.replace(path)


def train(arm, seed):
    api = source_imports()
    output = directory(arm, seed) / "complete.json"
    if output.exists():
        metadata = read(output)
        if (metadata["protocol"], metadata["arm"], metadata["seed"], metadata["budget"]) != (
                p.PROTOCOL, arm, seed, p.source_budget()):
            raise ValueError("existing source bundle has a different identity/budget")
        return
    run = SourceRun(arm, seed, api)
    checkpoint = p.RUN_ROOT / "source" / arm / f"seed_{seed}/checkpoints/train_state.pt"
    if checkpoint.exists():
        run.restore(api["torch"].load(checkpoint, map_location="cpu", weights_only=False))
    else:
        run.rollout(10000, -1, warmup=True)
    beginning = time.monotonic()
    for iteration in range(run.next_iteration, 2100):
        started = time.monotonic()
        run.rollout(p.source_samples(iteration), iteration)
        metrics = run.update()
        run.next_iteration = iteration + 1
        if iteration < 3 or run.next_iteration % 10 == 0:
            print(f"Iter {run.next_iteration}/2100 | steps={run.steps} | updates={run.trainer._n_train_steps_total} "
                  f"| seconds={time.monotonic()-started:.3f} | metrics={metrics}", flush=True)
        if run.next_iteration % 50 == 0 or run.next_iteration == 2100:
            run.save(checkpoint)
    budget = {"next_iteration": run.next_iteration, "total_steps": run.steps,
              "critic_calls": run.trainer._n_train_steps_total,
              "actor_opportunities": run.trainer._n_train_steps_total // 2}
    if budget != p.source_budget():
        raise ValueError("incomplete source comparison budget")
    graph, params = actor_from_torch(run.trainer.policy)
    common.save_pickle(directory(arm, seed) / "policy_params.pkl", common._to_numpy_tree(params))
    common.write_json(output, {"protocol": p.PROTOCOL, "arm": arm, "seed": seed,
        "budget": budget, "config": run.config, "runtime_versions": check_runtime(),
        "torch": api["torch"].__version__, "source": "unmodified bundled RESACTrainer/VectorizedQNetwork",
        "optimizer_counts": run.optimizer_counts, "sampling_device": "cpu", "learner_device": "cuda",
        "elapsed_this_attempt_seconds": time.monotonic()-beginning})


def audit(seed):
    world = common.World("hc", seed, physics_unroll=1)
    actors, budgets = {}, {}
    for arm in p.SOURCE_ARMS:
        model = GaussianPolicy(17, 6, 256, rngs=nnx.Rngs(seed))
        metadata = read(directory(arm, seed) / "complete.json")
        if ((metadata["protocol"], metadata["arm"], metadata["seed"]) != (p.PROTOCOL, arm, seed)
                or metadata["budget"] != p.source_budget() or metadata["runtime_versions"] != p.RUNTIME):
            raise ValueError("source controller runtime/budget differs")
        params = common._restore_tree_like(nnx.state(model, nnx.Param),
            common.read_pickle(directory(arm, seed) / "policy_params.pkl"), arm, allow_fallback=False)
        actors[arm], budgets[arm] = (nnx.graphdef(model), params), metadata
    model = GaussianPolicy(17, 6, 256, rngs=nnx.Rngs(seed))
    original = historical.bundle_dir("reference", seed)
    reference_metadata = read(original / "complete.json")
    if (reference_metadata["identity"] != historical.identity("reference", seed)
            or reference_metadata["budget"] != historical.budget("reference")
            or reference_metadata["runtime_versions"] != p.RUNTIME):
        raise ValueError("paired BAPR reference identity/runtime/budget differs")
    params = common._restore_tree_like(nnx.state(model, nnx.Param),
        common.read_pickle(original / "parameters.pkl")["policy"], "paired BAPR reference", allow_fallback=False)
    actors["reference"] = (nnx.graphdef(model), params)
    inverse, parameters, variance, unused = common.frozen_inverse()
    scores = common.inverse_evidence(inverse, parameters, world.gains, variance)
    routes = [("robust", "full", None, True), ("uncompensated", "no_shift", None, True),
              ("causal", "reference", scores, True), ("oracle", "reference", None, True)]
    events = common.evaluate_routes(world, actors, routes, event_seeds=p.EVENT_SEEDS, episodes=5)
    # Route names encode deployment semantics; publish source method names explicitly.
    events["source_full"] = events.pop("robust")
    events["source_no_shift"] = events.pop("uncompensated")
    stat = stationary(world, actors, routes)
    stat["source_full"], stat["source_no_shift"] = stat.pop("robust"), stat.pop("uncompensated")
    if any(row["steps"] != 1000 or row["switches_observed"] != 3 or row["termination"]
           for episodes in events.values() for row in episodes):
        raise ValueError("source comparison lost its switching windows")
    common.write_json(p.audit_path("source", seed), {"protocol": p.PROTOCOL, "suite": "source", "seed": seed,
        "runtime_versions": check_runtime(), "source_metadata": budgets,
        "reference_metadata": reference_metadata, "events": events, "stationary": stat,
        "means": {arm: float(np.mean([row["return"] for row in episodes])) for arm, episodes in events.items()}})


def analyze():
    rows = [read(p.audit_path("source", seed)) for seed in p.SOURCE_SEEDS]
    if any(row["seed"] != seed or row["protocol"] != p.PROTOCOL or row["runtime_versions"] != p.RUNTIME
           for row, seed in zip(rows, p.SOURCE_SEEDS, strict=True)):
        raise ValueError("mixed source audits")
    values = {arm: [row["means"][arm] for row in rows] for arm in rows[0]["means"]}
    common.write_json(p.analysis_path("source"), {"protocol": p.PROTOCOL, "policy_seeds": list(p.SOURCE_SEEDS),
        "means": {arm: float(np.mean(row)) for arm, row in values.items()}, "seed_means": values,
        "comparisons": {f"causal-minus-{arm}": common.paired_summary(values["causal"], values[arm])
                        for arm in ("source_full", "source_no_shift")},
        "scope": "Original Torch trainer/network on a common Brax benchmark, not original Gym-paper replication",
        "budgets": {"source": p.source_budget(), "bapr": historical.budget("reference")},
        "extra_estimator_cost_matched": False, "source_actor_period": 2, "bapr_actor_period": 1,
        "historical_cohorts_pooled": False})


def smoke():
    api = source_imports()
    torch = api["torch"]
    rows = {}
    for arm in p.SOURCE_ARMS:
        run = SourceRun(arm, p.SOURCE_SEEDS[0], api, capacity=4096)
        graph, params = actor_from_torch(run.trainer.policy)
        obs = np.random.default_rng(11).normal(size=(8, 17)).astype(np.float32)
        with torch.no_grad():
            expected = run.trainer.policy(torch.tensor(obs, device=api["ptu"].device), deterministic=True)[0].cpu().numpy()
        actual = np.asarray(nnx.merge(graph, params).deterministic(jnp.asarray(obs)))
        export_error = float(np.max(np.abs(expected-actual)))
        if not np.allclose(expected, actual, atol=1e-6, rtol=0):
            raise ValueError("Torch actor export changed deterministic actions")
        with torch.no_grad():
            outputs = run.trainer.policy(torch.tensor(obs, device=api["ptu"].device), deterministic=True)
        means = nnx.merge(graph, params)(jnp.asarray(obs))
        distribution_error = max(float(np.max(np.abs(np.asarray(left)-right.cpu().numpy())))
                                 for left, right in zip(means, outputs[1:3], strict=True))
        if distribution_error > 1e-6:
            raise ValueError("Torch actor export changed Gaussian distribution parameters")
        run.rollout(64, -1, warmup=True)
        run.update(4)
        checkpoint = p.RUN_ROOT / "smoke/source" / arm / "train_state.pt"
        run.save(checkpoint)
        run.update(2)
        expected = run.state()
        resumed = SourceRun(arm, p.SOURCE_SEEDS[0], api, capacity=4096)
        resumed.restore(torch.load(checkpoint, map_location="cpu", weights_only=False))
        resumed.update(2)
        actual = resumed.state()
        for left, right in zip(expected["models"], actual["models"], strict=True):
            if any(not torch.equal(left[key], right[key]) for key in left):
                raise ValueError("original trainer next-update resume differs")
        if not torch.equal(expected["log_alpha"], actual["log_alpha"]):
            raise ValueError("resumed entropy coefficient differs")
        metrics = resumed.update(2)
        if arm == "full" and metrics["Reg Norm Mean"] <= 0:
            raise ValueError("regularization no longer has its original positive sign")
        rows[arm] = {"export_max_error": export_error, "distribution_max_error": distribution_error,
                     "exact_next_update_resume": True, "metrics": metrics}
    common.write_json(p.smoke("source"), {"passed": True, "protocol": p.PROTOCOL,
        "runtime_versions": check_runtime(), "torch": torch.__version__, "arms": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train", "audit", "analyze"))
    parser.add_argument("--arm", choices=p.SOURCE_ARMS)
    parser.add_argument("--seed", type=int, choices=p.SOURCE_SEEDS)
    args = parser.parse_args()
    check_runtime()
    if jax.default_backend() != "cpu":
        raise ValueError("source bridge uses pinned CPU physics and Torch CUDA updates")
    if args.mode == "train":
        train(args.arm, args.seed)
    elif args.mode == "audit":
        audit(args.seed)
    else:
        globals()[args.mode]()
    print(f"SOURCE_COMPARISON_COMPLETE mode={args.mode}", flush=True)


if __name__ == "__main__":
    main()
