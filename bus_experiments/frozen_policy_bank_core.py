"""Resumable bus SAC training and compact policy bundles.

This module keeps the legacy bus reward, state, and RE-SAC regularization sign,
but separates robust-source and fixed-mode specialist training so the policy
headroom can be measured before fitting a mode estimator.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from env.sim import env_bus


CAT_COLS = ("bus_id", "station_id", "time_period", "direction")


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    max_episodes: int
    role: str
    fixed_mode: str | None = None
    route_sigma: float = 1.5
    mode_switch_min: int = 1800
    mode_switch_max: int = 7200
    hidden_dim: int = 64
    ensemble_size: int = 10
    replay_capacity: int = 1_000_000
    batch_size: int = 2048
    training_freq: int = 5
    critic_actor_ratio: int = 2
    gamma: float = 0.99
    soft_tau: float = 0.01
    reward_scale: float = 10.0
    weight_reg: float = 0.01
    beta: float = -2.0
    beta_ood: float = 0.01
    beta_bc: float = 0.001
    maximum_alpha: float = 0.6
    learning_rate: float = 1e-5
    checkpoint_interval: int = 10

    def __post_init__(self) -> None:
        if self.role not in {"robust_source", "specialist"}:
            raise ValueError(f"unsupported bus training role: {self.role}")
        if self.role == "specialist" and self.fixed_mode is None:
            raise ValueError("specialist training requires fixed_mode")
        if self.role == "robust_source" and self.fixed_mode is not None:
            raise ValueError("robust-source training cannot set fixed_mode")


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class ReplayBuffer:
    """Compact float32 ring buffer with a migratable on-disk snapshot."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.position = 0
        self.size = 0
        self.states: np.ndarray | None = None
        self.actions: np.ndarray | None = None
        self.rewards: np.ndarray | None = None
        self.next_states: np.ndarray | None = None
        self.dones: np.ndarray | None = None

    def _allocate(self, state: np.ndarray, action: np.ndarray) -> None:
        state_shape = tuple(np.asarray(state).shape)
        action_shape = tuple(np.asarray(action).shape)
        self.states = np.empty((self.capacity, *state_shape), dtype=np.float32)
        self.actions = np.empty((self.capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((self.capacity,), dtype=np.float32)
        self.next_states = np.empty(
            (self.capacity, *state_shape), dtype=np.float32)
        self.dones = np.empty((self.capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done) -> None:
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        if self.states is None:
            self._allocate(state, action)
        assert self.states is not None
        assert self.actions is not None
        assert self.rewards is not None
        assert self.next_states is not None
        assert self.dones is not None
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = float(reward)
        self.next_states[self.position] = np.asarray(next_state, dtype=np.float32)
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        if self.size < int(batch_size):
            raise ValueError("not enough replay entries")
        indices = np.random.randint(0, self.size, size=int(batch_size))
        assert self.states is not None
        assert self.actions is not None
        assert self.rewards is not None
        assert self.next_states is not None
        assert self.dones is not None
        return (
            self.states[indices], self.actions[indices],
            self.rewards[indices], self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size

    def save(self, path: Path) -> None:
        if self.states is None:
            payload = {
                "capacity": np.asarray(self.capacity, dtype=np.int64),
                "position": np.asarray(0, dtype=np.int64),
                "size": np.asarray(0, dtype=np.int64),
            }
        else:
            assert self.actions is not None
            assert self.rewards is not None
            assert self.next_states is not None
            assert self.dones is not None
            payload = {
                "capacity": np.asarray(self.capacity, dtype=np.int64),
                "position": np.asarray(self.position, dtype=np.int64),
                "size": np.asarray(self.size, dtype=np.int64),
                "states": self.states[:self.size],
                "actions": self.actions[:self.size],
                "rewards": self.rewards[:self.size],
                "next_states": self.next_states[:self.size],
                "dones": self.dones[:self.size],
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, **payload)
        temporary.replace(path)

    def load(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as payload:
            capacity = int(payload["capacity"])
            if capacity != self.capacity:
                raise ValueError(
                    f"replay capacity mismatch: {capacity} != {self.capacity}")
            size = int(payload["size"])
            self.size = size
            self.position = int(payload["position"])
            if size == 0:
                return
            states = np.asarray(payload["states"], dtype=np.float32)
            actions = np.asarray(payload["actions"], dtype=np.float32)
            self._allocate(states[0], actions[0])
            assert self.states is not None
            assert self.actions is not None
            assert self.rewards is not None
            assert self.next_states is not None
            assert self.dones is not None
            self.states[:size] = states
            self.actions[:size] = actions
            self.rewards[:size] = payload["rewards"]
            self.next_states[:size] = payload["next_states"]
            self.dones[:size] = payload["dones"]


class EmbeddingLayer(nn.Module):
    def __init__(self, category_sizes: tuple[int, ...]):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, min(50, size // 2))
            for size in category_sizes
        ])

    @property
    def output_dim(self) -> int:
        return sum(layer.embedding_dim for layer in self.embeddings)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            layer(values[:, index].long())
            for index, layer in enumerate(self.embeddings)
        ], dim=1)


class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(
            ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.weight:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values @ self.weight + self.bias


class BusCritic(nn.Module):
    def __init__(self, raw_state_dim: int, action_dim: int, hidden_dim: int,
                 ensemble_size: int, category_sizes: tuple[int, ...]):
        super().__init__()
        self.embedding = EmbeddingLayer(category_sizes)
        continuous_dim = raw_state_dim - len(category_sizes)
        input_dim = self.embedding.output_dim + continuous_dim + action_dim
        self.critic = nn.Sequential(
            VectorizedLinear(input_dim, hidden_dim, ensemble_size), nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, ensemble_size), nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, ensemble_size), nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, ensemble_size),
        )
        self.ensemble_size = int(ensemble_size)
        self.category_count = len(category_sizes)

    def forward(self, state: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        categorical = state[:, :self.category_count]
        continuous = state[:, self.category_count:]
        embedded = self.embedding(categorical)
        state_action = torch.cat([embedded, continuous, action], dim=-1)
        state_action = state_action.unsqueeze(0).repeat_interleave(
            self.ensemble_size, dim=0)
        return self.critic(state_action).squeeze(-1)

    def regularization_norm(self) -> torch.Tensor:
        layers = [
            module for module in self.critic
            if isinstance(module, VectorizedLinear)
        ]
        weight_norm = [
            torch.norm(layer.weight, p=1, dim=(1, 2)) for layer in layers]
        bias_norm = [
            torch.norm(layer.bias, p=1, dim=(1, 2)) for layer in layers[:-1]]
        return torch.stack(weight_norm).sum(0) + torch.stack(bias_norm).sum(0)


class BusPolicy(nn.Module):
    def __init__(self, raw_state_dim: int, action_dim: int, hidden_dim: int,
                 category_sizes: tuple[int, ...], action_range: float):
        super().__init__()
        self.embedding = EmbeddingLayer(category_sizes)
        continuous_dim = raw_state_dim - len(category_sizes)
        input_dim = self.embedding.output_dim + continuous_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.mean.weight.data.uniform_(-3e-3, 3e-3)
        self.mean.bias.data.uniform_(-3e-3, 3e-3)
        self.log_std.weight.data.uniform_(-3e-3, 3e-3)
        self.log_std.bias.data.uniform_(-3e-3, 3e-3)
        self.action_range = float(action_range)
        self.category_count = len(category_sizes)

    def forward(self, state: torch.Tensor):
        categorical = state[:, :self.category_count]
        continuous = state[:, self.category_count:]
        hidden = self.layers(torch.cat([
            self.embedding(categorical), continuous], dim=1))
        return self.mean(hidden), torch.clamp(self.log_std(hidden), -20, 2)

    def evaluate(self, state: torch.Tensor):
        mean, log_std = self(state)
        std = log_std.exp()
        noise = torch.randn_like(mean)
        pre_tanh = mean + std * noise
        unit_action = torch.tanh(pre_tanh)
        action = self.action_range * (unit_action + 1.0) / 2.0
        log_prob = (
            Normal(mean, std).log_prob(pre_tanh)
            - torch.log(1.0 - unit_action.pow(2) + 1e-6)
            - math.log(self.action_range)
        ).sum(dim=1)
        return action, log_prob

    @torch.no_grad()
    def get_action(self, state, deterministic: bool = True) -> np.ndarray:
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=next(self.parameters()).device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        mean, log_std = self(state_tensor)
        if deterministic:
            unit_action = torch.tanh(mean)
        else:
            unit_action = torch.tanh(mean + log_std.exp() * torch.randn_like(mean))
        action = self.action_range * (unit_action + 1.0) / 2.0
        return action[0].cpu().numpy()


class BusSAC:
    def __init__(self, environment: env_bus, config: TrainConfig,
                 device: torch.device):
        self.config = config
        self.device = device
        category_sizes = (
            int(environment.max_agent_num),
            int(round(len(environment.stations) / 2)),
            int(environment.timetables[-1].launch_time // 3600 + 2),
            2,
        )
        action_dim = int(environment.action_space.shape[0])
        action_range = float(environment.action_space.high[0])
        self.critic = BusCritic(
            environment.state_dim, action_dim, config.hidden_dim,
            config.ensemble_size, category_sizes).to(device)
        self.target_critic = deepcopy(self.critic).to(device)
        self.policy = BusPolicy(
            environment.state_dim, action_dim, config.hidden_dim,
            category_sizes, action_range).to(device)
        self.log_alpha = torch.zeros(
            1, dtype=torch.float32, requires_grad=True, device=device)
        self.alpha = 1.0
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.learning_rate)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config.learning_rate)
        self.action_dim = action_dim

    def select_action(self, state, deterministic: bool = False) -> np.ndarray:
        return self.policy.get_action(state, deterministic=deterministic)

    def update(self, replay: ReplayBuffer, update_index: int) -> dict[str, float]:
        cfg = self.config
        state, action, reward, next_state, done = replay.sample(cfg.batch_size)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        next_state_t = torch.as_tensor(
            next_state, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        reward_std = reward_t.std()
        if not torch.isfinite(reward_std) or reward_std < 1e-6:
            reward_std = torch.ones((), dtype=torch.float32, device=self.device)
        reward_t = cfg.reward_scale * (reward_t - reward_t.mean()) / reward_std

        new_action, log_prob = self.policy.evaluate(state_t)
        alpha_loss = -(
            self.log_alpha * (log_prob - self.action_dim).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = min(
            cfg.maximum_alpha, float(self.log_alpha.exp().detach().cpu()))

        with torch.no_grad():
            next_action, next_log_prob = self.policy.evaluate(next_state_t)
            reg_norm = self.target_critic.regularization_norm()
            target_q = self.target_critic(next_state_t, next_action)
            target_q = (
                target_q - self.alpha * next_log_prob.unsqueeze(0)
                + cfg.weight_reg * reg_norm.unsqueeze(1)
            )
            target = reward_t.unsqueeze(0) + (
                1.0 - done_t.unsqueeze(0)) * cfg.gamma * target_q

        predicted_q = self.critic(state_t, action_t)
        q_std = predicted_q.std(dim=0).mean()
        critic_loss = F.mse_loss(predicted_q, target) + cfg.beta_ood * q_std
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        policy_loss_value = float("nan")
        if update_index % cfg.critic_actor_ratio == 0:
            with torch.no_grad():
                actor_reg_norm = self.target_critic.regularization_norm()
            q_distribution = (
                self.critic(state_t, new_action)
                + cfg.weight_reg * actor_reg_norm.unsqueeze(1)
                - self.alpha * log_prob.unsqueeze(0)
            )
            q_mean = q_distribution.mean(dim=0)
            q_uncertainty = q_distribution.std(dim=0)
            q_objective = -(q_mean + cfg.beta * q_uncertainty).mean()
            policy_loss = q_objective + cfg.beta_bc * F.mse_loss(
                new_action, action_t)
            self.policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            self.policy_optimizer.step()
            policy_loss_value = float(policy_loss.detach().cpu())

        with torch.no_grad():
            for target_parameter, parameter in zip(
                    self.target_critic.parameters(), self.critic.parameters()):
                target_parameter.mul_(1.0 - cfg.soft_tau)
                target_parameter.add_(parameter, alpha=cfg.soft_tau)

        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "policy_loss": policy_loss_value,
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(self.alpha),
            "q_mean": float(predicted_q.mean().detach().cpu()),
            "q_std": float(q_std.detach().cpu()),
            "weighted_reg_mean": float(
                (cfg.weight_reg * reg_norm).mean().detach().cpu()),
        }

    def controller_state(self) -> dict[str, Any]:
        return {
            "schema": "bapr.bus-controller.v1",
            "config": asdict(self.config),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "policy": self.policy.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha": float(self.alpha),
        }

    def load_controller_state(self, payload: dict[str, Any]) -> None:
        if payload.get("schema") != "bapr.bus-controller.v1":
            raise ValueError("unsupported bus controller bundle")
        self.critic.load_state_dict(payload["critic"])
        self.target_critic.load_state_dict(payload["target_critic"])
        self.policy.load_state_dict(payload["policy"])
        with torch.no_grad():
            self.log_alpha.copy_(payload["log_alpha"].to(self.device))
        self.alpha = float(payload["alpha"])


def make_environment(config: TrainConfig) -> env_bus:
    root = Path(__file__).resolve().parents[1]
    env_path = root / "env"
    if config.role == "specialist":
        return env_bus(
            str(env_path), route_sigma=config.route_sigma,
            fixed_mode=config.fixed_mode)
    return env_bus(
        str(env_path), route_sigma=config.route_sigma,
        enable_mode_switch=True,
        mode_switch_interval=(
            config.mode_switch_min, config.mode_switch_max),
        random_initial_mode=True,
    )


def _save_torch_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_training_checkpoint(
        checkpoint_dir: Path, trainer: BusSAC, replay: ReplayBuffer,
        completed_episodes: int, total_steps: int, last_trained_step: int,
        update_index: int, rewards: list[float], diagnostics: list[dict]) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot = f"episode_{int(completed_episodes):04d}"
    replay_path = checkpoint_dir / f"replay_{snapshot}.npz"
    state_path = checkpoint_dir / f"state_{snapshot}.pt"
    previous = None
    latest_path = checkpoint_dir / "latest.json"
    if latest_path.is_file():
        previous = json.loads(latest_path.read_text(encoding="utf-8"))
    replay.save(replay_path)
    payload = {
        "schema": "bapr.bus-training-state.v1",
        "config": asdict(trainer.config),
        "completed_episodes": int(completed_episodes),
        "total_steps": int(total_steps),
        "last_trained_step": int(last_trained_step),
        "update_index": int(update_index),
        "replay_file": replay_path.name,
        "rewards": [float(value) for value in rewards],
        "diagnostics": diagnostics,
        "controller": trainer.controller_state(),
        "critic_optimizer": trainer.critic_optimizer.state_dict(),
        "policy_optimizer": trainer.policy_optimizer.state_dict(),
        "alpha_optimizer": trainer.alpha_optimizer.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    _save_torch_atomic(state_path, payload)
    atomic_json(latest_path, {
        "schema": "bapr.bus-checkpoint-pointer.v1",
        "completed_episodes": int(completed_episodes),
        "state_file": state_path.name,
        "replay_file": replay_path.name,
    })
    if previous:
        for key in ("state_file", "replay_file"):
            old_path = checkpoint_dir / str(previous.get(key, ""))
            if old_path not in {state_path, replay_path} and old_path.is_file():
                old_path.unlink()


def load_training_checkpoint(
        checkpoint_dir: Path, trainer: BusSAC,
        replay: ReplayBuffer) -> dict[str, Any]:
    pointer = json.loads(
        (checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    if pointer.get("schema") != "bapr.bus-checkpoint-pointer.v1":
        raise ValueError("unsupported bus checkpoint pointer")
    payload = torch.load(
        checkpoint_dir / str(pointer["state_file"]),
        map_location=trainer.device, weights_only=False)
    if payload.get("schema") != "bapr.bus-training-state.v1":
        raise ValueError("unsupported bus training checkpoint")
    saved_config = dict(payload["config"])
    current_config = asdict(trainer.config)
    if saved_config != current_config:
        raise ValueError("bus checkpoint config does not match this task")
    trainer.load_controller_state(payload["controller"])
    trainer.critic_optimizer.load_state_dict(payload["critic_optimizer"])
    trainer.policy_optimizer.load_state_dict(payload["policy_optimizer"])
    trainer.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
    if payload.get("replay_file") != pointer.get("replay_file"):
        raise ValueError("bus checkpoint state/replay pointer mismatch")
    replay.load(checkpoint_dir / str(pointer["replay_file"]))
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"].cpu())
    if torch.cuda.is_available() and payload.get("cuda_rng"):
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload


def load_warmstart(path: Path, trainer: BusSAC) -> None:
    payload = torch.load(path, map_location=trainer.device, weights_only=False)
    trainer.load_controller_state(payload)


def publish_bundle(
        bundle_dir: Path, trainer: BusSAC, completed_episodes: int,
        total_steps: int, rewards: list[float], diagnostics: list[dict],
        warmstart_path: Path | None) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    controller_path = bundle_dir / "controller.pt"
    _save_torch_atomic(controller_path, trainer.controller_state())
    recent = rewards[-20:]
    summary = {
        "schema": "bapr.bus-train-summary.v1",
        "role": trainer.config.role,
        "seed": int(trainer.config.seed),
        "fixed_mode": trainer.config.fixed_mode,
        "completed_episodes": int(completed_episodes),
        "total_steps": int(total_steps),
        "reward_final": float(rewards[-1]) if rewards else None,
        "reward_recent_mean": float(np.mean(recent)) if recent else None,
        "reward_recent_std": float(np.std(recent)) if recent else None,
        "warmstart_path": str(warmstart_path) if warmstart_path else None,
        "config": asdict(trainer.config),
        "diagnostics_tail": diagnostics[-20:],
    }
    atomic_json(bundle_dir / "train_summary.json", summary)
    manifest = {
        "schema": "bapr.bus-controller-bundle.v1",
        "controller": "controller.pt",
        "train_summary": "train_summary.json",
        "complete": completed_episodes == trainer.config.max_episodes,
        "completed_episodes": int(completed_episodes),
    }
    atomic_json(bundle_dir / "bundle_manifest.json", manifest)


def train(
        config: TrainConfig, run_dir: Path, bundle_dir: Path, *,
        resume: bool, warmstart_path: Path | None = None) -> None:
    seed_all(config.seed)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("bus policy-bank training requires a CUDA device")
    environment = make_environment(config)
    replay = ReplayBuffer(config.replay_capacity)
    trainer = BusSAC(environment, config, device)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / "latest.json"

    completed_episodes = 0
    total_steps = 0
    last_trained_step = -1
    update_index = 0
    rewards: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    if resume and checkpoint_path.is_file():
        state = load_training_checkpoint(checkpoint_dir, trainer, replay)
        completed_episodes = int(state["completed_episodes"])
        total_steps = int(state["total_steps"])
        last_trained_step = int(state["last_trained_step"])
        update_index = int(state["update_index"])
        rewards = [float(value) for value in state["rewards"]]
        diagnostics = list(state["diagnostics"])
        print(
            f"Resumed bus task at episode {completed_episodes}/"
            f"{config.max_episodes}, steps={total_steps}, replay={len(replay)}",
            flush=True)
    elif warmstart_path is not None:
        load_warmstart(warmstart_path, trainer)
        print(f"Loaded frozen robust warmstart: {warmstart_path}", flush=True)

    if completed_episodes > config.max_episodes:
        raise ValueError("checkpoint exceeds requested episode budget")

    for episode in range(completed_episodes, config.max_episodes):
        started_at = time.time()
        environment.reset()
        state_dict, reward_dict, _ = environment.initialize_state(render=False)
        action_dict = {
            key: None for key in range(environment.max_agent_num)}
        episode_reward = 0.0
        episode_steps = 0
        update_rows: list[dict[str, float]] = []
        done = False

        while not done:
            for key in state_dict:
                agent_states = state_dict[key]
                if len(agent_states) == 1:
                    if action_dict[key] is None:
                        action_dict[key] = trainer.select_action(
                            np.asarray(agent_states[0]), deterministic=False)
                elif len(agent_states) == 2:
                    if agent_states[0][1] != agent_states[1][1]:
                        state = np.asarray(agent_states[0], dtype=np.float32)
                        next_state = np.asarray(agent_states[1], dtype=np.float32)
                        reward = float(reward_dict[key])
                        replay.push(
                            state, action_dict[key], reward, next_state, False)
                        episode_reward += reward
                        episode_steps += 1
                        total_steps += 1
                    state_dict[key] = agent_states[1:]
                    action_dict[key] = trainer.select_action(
                        np.asarray(state_dict[key][0]), deterministic=False)

            state_dict, reward_dict, done = environment.step(action_dict)
            if (
                    len(replay) > config.batch_size
                    and len(replay) % config.training_freq == 0
                    and last_trained_step != total_steps):
                last_trained_step = total_steps
                update_rows.append(trainer.update(replay, update_index))
                update_index += 1

        rewards.append(float(episode_reward))
        duration = time.time() - started_at
        row: dict[str, Any] = {
            "episode": int(episode),
            "reward": float(episode_reward),
            "duration_s": float(duration),
            "episode_steps": int(episode_steps),
            "total_steps": int(total_steps),
            "updates": int(len(update_rows)),
            "mode_switches": int(environment.mode_switch_count),
            "final_mode": str(environment.current_mode_name),
        }
        for key in (
                "critic_loss", "policy_loss", "alpha", "q_mean", "q_std",
                "weighted_reg_mean"):
            values = [entry[key] for entry in update_rows
                      if np.isfinite(entry.get(key, np.nan))]
            row[key] = float(np.mean(values)) if values else None
        diagnostics.append(row)
        completed_episodes = episode + 1
        print(
            f"Episode {completed_episodes}/{config.max_episodes} | "
            f"reward={episode_reward:.1f} | steps={total_steps} | "
            f"replay={len(replay)} | mode={environment.current_mode_name} | "
            f"switches={environment.mode_switch_count} | {duration:.1f}s/episode",
            flush=True)

        if (
                completed_episodes % config.checkpoint_interval == 0
                or completed_episodes == config.max_episodes):
            save_training_checkpoint(
                checkpoint_dir, trainer, replay, completed_episodes,
                total_steps, last_trained_step, update_index, rewards,
                diagnostics)
            atomic_json(run_dir / "progress.json", row)

    publish_bundle(
        bundle_dir, trainer, completed_episodes, total_steps, rewards,
        diagnostics, warmstart_path)
    print(
        f"BUS_POLICY_BANK_TRAINING_COMPLETE episodes={completed_episodes} "
        f"steps={total_steps} bundle={bundle_dir}", flush=True)


def load_controller(
        controller_path: Path, environment: env_bus,
        device: torch.device | None = None) -> BusSAC:
    device = device or torch.device("cpu")
    payload = torch.load(controller_path, map_location=device, weights_only=False)
    config = TrainConfig(**payload["config"])
    trainer = BusSAC(environment, config, device)
    trainer.load_controller_state(payload)
    trainer.policy.eval()
    trainer.critic.eval()
    trainer.target_critic.eval()
    return trainer
