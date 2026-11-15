"""Use the untouched Torch trainer and rlkit collectors in their native Gym world."""
import argparse
from copy import deepcopy
import json
import math
import os
import random
import sys
import time

from jax_experiments.analysis import resac_torch_native_protocol_v43 as p


def imports():
    sys.path[:0] = [str(p.SOURCE)]
    import gym_patch
    import gym
    import mujoco_py
    import numpy as np
    import torch
    from resac_networks import VectorizedQNetwork
    from resac_trainer import RESACTrainer
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.torch_replay_buffer import TorchReplayBuffer
    from rlkit.envs import make_env
    from rlkit.envs.vecenv import VectorEnv
    from rlkit.samplers.data_collector import VecMdpPathCollector, VecMdpStepCollector
    from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
    runtime = {"torch": torch.__version__, "gym": gym.__version__, "numpy": np.__version__,
               "mujoco_py": mujoco_py.__version__, "mujoco": mujoco_py.functions.mj_version()}
    if runtime != p.RUNTIME:
        raise ValueError(f"native runtime mismatch: {runtime}")
    return locals()


def cpu_tree(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item, torch) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item, torch) for item in value)
    return deepcopy(value)


class NativeRun:
    def __init__(self, arm, seed, api, gpu=True, protocol=p):
        self.api, self.arm, self.seed = api, arm, seed
        self.protocol = protocol
        self.config = protocol.variant(arm)
        torch, np, gym, ptu = [api[name] for name in ("torch", "np", "gym", "ptu")]
        torch.set_num_threads(4)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        gym.spaces.np_random.seed(seed)
        ptu.set_gpu_mode(gpu, 0)
        if gpu:
            torch.cuda.manual_seed(seed)
        dummy = api["make_env"](self.config["env"])
        self.expl = api["VectorEnv"]([lambda: api["make_env"](self.config["env"])] * 4)
        self.eval = api["VectorEnv"]([lambda: api["make_env"](self.config["env"])] * 4)
        self.expl.seed(seed)
        self.eval.seed(seed)
        critic = api["VectorizedQNetwork"](dummy.observation_space.low.size,
            dummy.action_space.low.size, self.config["layer_size"], self.config["ensemble_size"])
        policy = api["TanhGaussianPolicy"]([256, 256], dummy.observation_space.low.size,
                                           dummy.action_space.low.size)
        self.trainer = api["RESACTrainer"](dummy, policy, critic, deepcopy(critic),
            ensemble_size=self.config["ensemble_size"], **self.config["trainer_kwargs"])
        for model in self.trainer.networks:
            model.to(ptu.device)
        self.replay = api["TorchReplayBuffer"](self.config["replay_buffer_size"], dummy)
        self.collector = api["VecMdpStepCollector"](self.expl, policy)
        self.evaluator = api["VecMdpPathCollector"](self.eval, api["MakeDeterministic"](policy))
        self.next_epoch, self.total_steps, self.train_data = 0, 0, None
        self.rows, self.optimizer_counts = [], {"critic": 0, "actor": 0}
        for name, optimizer in (("critic", self.trainer.qf_optimizer), ("actor", self.trainer.policy_optimizer)):
            optimizer.register_step_post_hook(lambda opt, args, kwargs, key=name: self.count_step(key))
        self.physics = {name: getattr(dummy.unwrapped.model, name).tolist()
                        for name in ("body_mass", "dof_damping")}
        self.physics["gravity"] = dummy.unwrapped.model.opt.gravity.tolist()
        self.physics["dt"] = dummy.unwrapped.dt
        dummy.close()

    def count_step(self, name):
        self.optimizer_counts[name] += 1

    def warmup(self):
        paths = self.collector.collect_new_steps(1000, 2500, False, random=True)
        self.replay.add_paths(paths)
        self.total_steps = 10000
        self.collector.end_epoch(-1)
        self.train_data = self.replay.next_batch(256)

    def vector_step(self):
        for model in self.trainer.networks:
            model.train(False)
        self.replay.add_paths(self.collector.collect_new_steps(1000, 1, False))
        self.total_steps += 4
        for model in self.trainer.networks:
            model.train(True)
        for _ in range(4):
            self.trainer.train(self.train_data)
            self.train_data = self.replay.next_batch(256)

    def epoch(self):
        torch, np = self.api["torch"], self.api["np"]
        start = time.monotonic()
        for _ in range(250):
            self.vector_step()
        torch.cuda.synchronize()
        update_seconds = time.monotonic() - start
        for model in self.trainer.networks:
            model.train(False)
        self.evaluator.collect_new_paths(1000, 5, True)
        paths = list(self.evaluator.get_epoch_paths())
        evaluation = {"returns": [float(np.sum(path["rewards"])) for path in paths],
                      "lengths": [len(path["actions"]) for path in paths],
                      "terminated": [bool(path["terminals"][-1, 0]) for path in paths]}
        metrics = {name: float(value) for name, value in self.trainer.get_diagnostics().items()}
        if not all(math.isfinite(value) for value in [*metrics.values(), *evaluation["returns"]]):
            raise ValueError("nonfinite native training/evaluation")
        row = {"epoch": self.next_epoch, "evaluation": evaluation, "metrics": metrics,
               "update_seconds": update_seconds, "epoch_seconds": time.monotonic() - start,
               "optimizer_steps": dict(self.optimizer_counts), "amp_scale": self.trainer.scaler.get_scale()}
        self.rows.append(row)
        self.collector.end_epoch(self.next_epoch)
        self.evaluator.end_epoch(self.next_epoch)
        self.trainer.end_epoch(self.next_epoch)
        self.next_epoch += 1
        print(f"Epoch {self.next_epoch}/1000 | steps={self.total_steps} | updates={self.trainer._n_train_steps_total} "
              f"| return={np.mean(evaluation['returns']):.2f} | {row['epoch_seconds']:.2f}s/epoch", flush=True)
        return row

    def state(self):
        torch, np, gym = [self.api[name] for name in ("torch", "np", "gym")]
        trainer = self.trainer
        replay_fields = ("_observations", "_next_obs", "_actions", "_rewards", "_terminals", "_env_infos")
        return {"protocol": self.protocol.PROTOCOL, "arm": self.arm, "seed": self.seed,
            "variant": self.config, "runtime": self.protocol.RUNTIME, "next_epoch": self.next_epoch,
            "total_steps": self.total_steps, "rows": self.rows,
            "models": [cpu_tree(model.state_dict(), torch) for model in trainer.networks],
            "optimizers": [cpu_tree(opt.state_dict(), torch) for opt in
                (trainer.qf_optimizer, trainer.policy_optimizer, trainer.alpha_optimizer)],
            "log_alpha": cpu_tree(trainer.log_alpha, torch), "scaler": trainer.scaler.state_dict(),
            "critic_calls": trainer._n_train_steps_total, "trainer_calls": trainer._num_train_steps,
            "optimizer_counts": dict(self.optimizer_counts),
            "train_data": cpu_tree(self.train_data, torch), "prefetch": cpu_tree(self.replay.batch, torch),
            "replay": {key: cpu_tree(getattr(self.replay, key), torch) for key in replay_fields},
            "replay_top": self.replay._top, "replay_size": self.replay._size,
            "collector": {key: deepcopy(value) for key, value in vars(self.collector).items()
                          if key not in ("_env", "_policy")},
            "envs": [[self.env_state(env) for env in vector.envs] for vector in (self.expl, self.eval)],
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "spaces": gym.spaces.np_random.get_state(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state()}}

    @staticmethod
    def env_state(env):
        base = env.unwrapped
        return {"sim": base.sim.get_state(), "warmstart": base.sim.data.qacc_warmstart.copy(),
                "ctrl": base.sim.data.ctrl.copy(), "rng": base.np_random.get_state()}

    def restore(self, state):
        torch, np, gym, ptu = [self.api[name] for name in ("torch", "np", "gym", "ptu")]
        if (state["protocol"] != self.protocol.PROTOCOL or state["arm"] != self.arm or state["seed"] != self.seed
                or state["variant"] != self.config or state["runtime"] != self.protocol.RUNTIME):
            raise ValueError("checkpoint belongs to another native arm/runtime")
        trainer = self.trainer
        for model, parameters in zip(trainer.networks, state["models"]):
            model.load_state_dict(parameters)
        trainer.log_alpha.data.copy_(state["log_alpha"].to(ptu.device))
        for opt, value in zip((trainer.qf_optimizer, trainer.policy_optimizer, trainer.alpha_optimizer), state["optimizers"]):
            opt.load_state_dict(value)
        trainer.scaler.load_state_dict(state["scaler"])
        trainer._n_train_steps_total, trainer._cached_reg_norm = state["critic_calls"], None
        trainer._num_train_steps = state["trainer_calls"]
        trainer.end_epoch(state["next_epoch"])
        self.optimizer_counts = dict(state["optimizer_counts"])
        for key, value in state["replay"].items():
            target = getattr(self.replay, key)
            if isinstance(value, dict):
                for name, tensor in value.items():
                    target[name].copy_(tensor)
            else:
                target.copy_(value)
        self.replay._top, self.replay._size = state["replay_top"], state["replay_size"]
        self.train_data = {key: value.to(ptu.device) for key, value in state["train_data"].items()}
        self.replay.batch = {key: value.to(ptu.device) for key, value in state["prefetch"].items()}
        vars(self.collector).update(deepcopy(state["collector"]))
        for vector, env_states in zip((self.expl, self.eval), state["envs"]):
            for env, value in zip(vector.envs, env_states):
                base = env.unwrapped
                base.sim.set_state(value["sim"])
                base.sim.forward()
                base.sim.data.qacc_warmstart[:] = value["warmstart"]
                base.sim.data.ctrl[:] = value["ctrl"]
                base.np_random.set_state(value["rng"])
            vector._obs = np.stack([env.unwrapped._get_obs() for env in vector.envs])
        self.next_epoch, self.total_steps, self.rows = state["next_epoch"], state["total_steps"], state["rows"]
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        gym.spaces.np_random.set_state(rng["spaces"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"])
        print(f"RESUME epoch={self.next_epoch} steps={self.total_steps} updates={trainer._n_train_steps_total}", flush=True)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        self.api["torch"].save(self.state(), temporary)
        temporary.replace(path)

    def close(self):
        self.expl.close()
        self.eval.close()


def smoke(api, protocol=p, run_class=NativeRun):
    rows = {}
    torch, np = api["torch"], api["np"]
    for arm in protocol.ARMS:
        run = run_class(arm, protocol.SEEDS[0], api, protocol=protocol)
        run.warmup()
        run.epoch()
        run.epoch()
        checkpoint = protocol.RUN_ROOT / "smoke" / arm / "train_state.pt"
        run.save(checkpoint)
        run.vector_step()
        expected = run.state()
        run.close()
        del run
        resumed = run_class(arm, protocol.SEEDS[0], api, protocol=protocol)
        resumed.restore(torch.load(checkpoint, map_location="cpu", weights_only=False))
        resumed.vector_step()
        actual = resumed.state()
        for wanted, got in zip(expected["models"], actual["models"]):
            for key in wanted:
                torch.testing.assert_close(got[key], wanted[key], rtol=1e-6, atol=1e-7)
        for wanted, got in zip(expected["optimizers"], actual["optimizers"]):
            for parameter, values in wanted["state"].items():
                for key, value in values.items():
                    torch.testing.assert_close(got["state"][parameter][key], value, rtol=1e-6, atol=1e-7)
        for field in ("total_steps", "critic_calls", "optimizer_counts", "scaler"):
            assert expected[field] == actual[field], field
        torch.testing.assert_close(actual["log_alpha"], expected["log_alpha"], rtol=1e-6, atol=1e-7)
        for field in ("train_data", "prefetch"):
            for key in expected[field]:
                torch.testing.assert_close(actual[field][key], expected[field][key], rtol=0, atol=0)
        np.testing.assert_allclose(actual["collector"]["_obs"], expected["collector"]["_obs"], rtol=1e-10, atol=1e-10)
        rows[arm] = {"epoch_seconds": resumed.rows[-1]["epoch_seconds"],
                     "estimated_training_hours": resumed.rows[-1]["epoch_seconds"] * 1000 / 3600,
                     "resume_vector_step_and_four_updates": "passed", "physics": resumed.physics,
                     "metrics": resumed.rows[-1]["metrics"], "optimizer_steps": actual["optimizer_counts"]}
        resumed.close()
        del resumed
    record = {"protocol": protocol.PROTOCOL, "runtime": protocol.RUNTIME, "arms": rows,
              "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20}
    protocol.write(protocol.COMPACT / "smoke/result.json", record)
    if all(row["estimated_training_hours"] <= 20 for row in rows.values()):
        protocol.write(protocol.SMOKE, record)


def train(arm, seed, api, protocol=p, run_class=NativeRun):
    if protocol.result(arm, seed).exists():
        protocol.validate(json.loads(protocol.result(arm, seed).read_text()), arm, seed)
        return
    run = run_class(arm, seed, api, protocol=protocol)
    checkpoint = protocol.directory(arm, seed) / "checkpoints/train_state.pt"
    if checkpoint.exists():
        run.restore(api["torch"].load(checkpoint, map_location="cpu", weights_only=False))
    else:
        run.warmup()
        run.save(checkpoint)
    while run.next_epoch < 1000:
        run.epoch()
        if run.next_epoch % 25 == 0:
            run.save(checkpoint)
    row = {"protocol": protocol.PROTOCOL, "arm": arm, "seed": seed, "runtime": protocol.RUNTIME,
        "variant": run.config, "physics": run.physics,
        "budget": {"next_epoch": run.next_epoch, "total_steps": run.total_steps,
                   "critic_calls": run.trainer._n_train_steps_total,
                   "actor_opportunities": (run.trainer._n_train_steps_total + 1) // 2},
        "optimizer_steps": run.optimizer_counts, "final_evaluation": run.rows[-1]["evaluation"],
        "timeline": [item for item in run.rows if item["epoch"] in (0, 1, 9, 49, 99, 249, 499, 999)]}
    protocol.validate(row, arm, seed)
    protocol.write(protocol.result(arm, seed), row)
    run.close()


def analyze():
    from statistics import mean, stdev
    rows = {arm: [json.loads(p.result(arm, seed).read_text()) for seed in p.SEEDS] for arm in p.ARMS}
    for arm, values in rows.items():
        for seed, row in zip(p.SEEDS, values):
            p.validate(row, arm, seed)
    returns = {arm: [mean(row["final_evaluation"]["returns"]) for row in values] for arm, values in rows.items()}
    delta = [left - right for left, right in zip(returns["full"], returns["no_shift"])]
    t95_df2 = math.sqrt(2 * 0.95**2 / (1 - 0.95**2))
    radius = t95_df2 * stdev(delta) / math.sqrt(3)
    p.write(p.COMPACT / "analysis/analysis.json", {"protocol": p.PROTOCOL, "seeds": p.SEEDS,
        "returns": returns, "full_minus_zero": {"mean": mean(delta), "ci95": [mean(delta)-radius, mean(delta)+radius],
        "per_seed": delta, "wins": sum(value > 0 for value in delta)}, "runs": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train", "analyze"))
    parser.add_argument("--arm", choices=p.ARMS)
    parser.add_argument("--seed", type=int, choices=p.SEEDS)
    args = parser.parse_args()
    if args.mode == "analyze":
        analyze()
    else:
        api = imports()
        (smoke(api) if args.mode == "smoke" else train(args.arm, args.seed, api))


if __name__ == "__main__":
    main()
