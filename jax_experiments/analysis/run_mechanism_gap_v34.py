"""Causal identification ablations, information-matched SAC, and Reacher screen.

Full replay/checkpoints stay on the training node. Downstream jobs consume only
small actor/estimator bundles and per-seed JSON audits.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from functools import partial
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import envs
from flax import nnx
from scipy.stats import t as student_t

from jax_experiments.algos.sac_base import SACBase
from jax_experiments.analysis import mechanism_gap_protocol_v34 as p
from jax_experiments.analysis.regime_polarity_posterior import FilterConfig
from jax_experiments.common.checkpoint import (
    _patch_flax_variablestate_unpickle, _restore_tree_like, _to_numpy_tree,
    load_checkpoint, save_checkpoint,
)
from jax_experiments.common.logging import Logger
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.configs.default import Config
from jax_experiments.envs.brax_env import apply_action_disturbance
from jax_experiments.envs.stochastic_mode_env import StochasticModeEnv
from jax_experiments.networks.executed_action_inverse import (
    ExecutedActionInverse, candidate_action_evidence,
)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_pickle(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def read_pickle(path):
    _patch_flax_variablestate_unpickle()
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def config(kind, seed):
    return Config(
        env_name="HalfCheetah-v2" if kind == "hc" else "reacher",
        brax_backend="spring" if kind == "hc" else "generalized",
        seed=seed, algo="sac", hidden_dim=256, ensemble_size=2,
        replay_size=1000000, batch_size=p.BATCH_SIZE, lr=3e-4,
        samples_per_iter=p.SAMPLES, updates_per_iter=p.UPDATES,
        initial_random_steps=0, start_train_steps=0,
    )


class World:
    """Native simulator transitions, with a separate persistent action map."""

    def __init__(self, kind, seed, *, physics_unroll=None, transforms=None):
        self.kind = kind
        # GPU timing preserves full traces; keep the native substep count.
        if physics_unroll is None:
            physics_unroll = 4 if kind == "hc" else 1
        if kind == "hc":
            self.env = StochasticModeEnv(
                "HalfCheetah-v2", family="actuator_polarity",
                backend="spring", dwell_steps=250, seed=seed)
            self.obs_dim, self.act_dim = self.env.obs_dim, self.env.act_dim
            self.gains = np.asarray([task["action_gain"] for task in self.env._tasks])
            self.reset = self.env._reset_state
            self.sys = self.env.base_sys

            def step(state, action):
                old = state.pipeline_state
                new = self.env._physics_step(self.sys, old, action)
                reward, observation, done = self.env._reward_obs(old, new, action)
                return state.replace(
                    pipeline_state=new, obs=observation, reward=reward, done=done)

            self.step = step
            self.horizon, self.dwell = 1000, 250
        else:
            self.env = envs.get_environment("reacher", backend="generalized")
            self.obs_dim = self.env.observation_size
            self.act_dim = self.env.action_size
            self.gains = np.asarray(((1, 1), (-1, 1), (1, -1), (-1, -1)), np.float32)
            self.reset = lambda unused, key: self.env.reset(key)
            self.step = self.env.step
            self.sys = self.env.sys
            self.horizon, self.dwell = p.REACHER_HORIZON, p.REACHER_DWELL
        if transforms is not None:
            self.gains = np.asarray(transforms, np.float32)
            if self.gains.shape != (4, self.act_dim, self.act_dim):
                raise ValueError("actuator transforms must match the four-mode action space")
            if (not np.isin(self.gains, (-1, 0, 1)).all()
                    or not np.array_equal(self.gains @ self.gains.transpose(0, 2, 1),
                                          np.broadcast_to(np.eye(self.act_dim), self.gains.shape))):
                raise ValueError("this experiment requires signed-permutation transforms")
        self.gains = jnp.asarray(self.gains, dtype=jnp.float32)
        if physics_unroll != 1:
            native = self.env.env if kind == "hc" else self.env
            pipeline, frames, debug = native._pipeline, native._n_frames, native._debug

            def physics_step(sys, state, action):
                def frame(current, unused):
                    return pipeline.step(sys, current, action, debug), None
                return jax.lax.scan(frame, state, (), frames, unroll=physics_unroll)[0]

            if kind == "hc":
                self.env._physics_step = physics_step
            else:
                self.env.pipeline_step = lambda state, action: physics_step(self.sys, state, action)


def nominal_signal(command, transform):
    return transform @ command if transform.ndim == 2 else command * transform


def compensated_command(nominal, gains, mode):
    if gains.ndim == 3:
        return gains[mode].T @ (gains[0] @ nominal)
    return nominal * gains[0] * gains[mode]


def update_belief(belief, evidence, controls, temporal=True):
    """One completed transition updates the posterior used at the next action."""
    if temporal:
        hazard, scale, decay = controls
        prior = (1 - hazard) * belief + hazard / (belief.shape[-1] - 1) * (1 - belief)
        logits = decay * jnp.log(jnp.clip(prior, 1e-12, 1)) + scale * evidence
    else:
        logits = controls[1] * evidence
    return jax.nn.softmax(logits)


def classifier_features(observation, action, next_observation):
    return jnp.concatenate((observation / 5, action, (next_observation - observation) / 0.1), axis=-1)


class DirectClassifier(nnx.Module):
    def __init__(self, obs_dim, act_dim, seed):
        rngs = nnx.Rngs(seed)
        self.first = nnx.Linear(2 * obs_dim + act_dim, 256, rngs=rngs)
        self.second = nnx.Linear(256, 256, rngs=rngs)
        self.output = nnx.Linear(256, 4, rngs=rngs)

    def __call__(self, observation, action, next_observation):
        x = classifier_features(observation, action, next_observation)
        return self.output(nnx.relu(self.second(nnx.relu(self.first(x)))))


def inverse_model(obs_dim, act_dim, seed):
    return ExecutedActionInverse(
        obs_dim, act_dim, hidden_dim=256, ensemble_size=5,
        n_layers=3, obs_scale=5, delta_scale=0.1, rngs=nnx.Rngs(seed))


def frozen_inverse():
    directory = p.RELEASE / "estimator"
    manifest = json.loads((directory / "model_manifest.json").read_text())
    model = inverse_model(17, 6, manifest["model_seed"])
    template = nnx.state(model, nnx.Param)
    leaves, tree = jax.tree.flatten(template)
    with np.load(directory / "inverse_params.npz") as archive:
        saved = [jnp.asarray(archive[row["key"]]) for row in manifest["parameter_leaves"]]
    if [value.shape for value in leaves] != [value.shape for value in saved]:
        raise ValueError("frozen inverse parameter layout differs from the runtime model")
    params = jax.tree.unflatten(tree, saved)
    controls = FilterConfig.from_dict(manifest["filter_config"])
    return model, params, jnp.asarray(manifest["residual_variance"]), (
        controls.hazard_rate, controls.evidence_scale, controls.posterior_decay)


def inverse_evidence(model, params, gains, variance, single_head=False):
    graph = nnx.graphdef(model)

    @jax.jit
    def evidence(observation, action, next_observation):
        predicted = nnx.merge(graph, params).predict(observation[None], next_observation[None])
        if single_head:
            predicted = predicted[:1]
        return candidate_action_evidence(predicted, action[None], gains, variance)[0][0]

    return evidence


def load_frozen_actor(seed):
    directory = p.RELEASE / "reference_policies" / f"seed_{seed}" / "mode_0"
    agent = SACBase(17, 6, config("hc", seed), seed)
    path = directory / "policy/policy_params.pkl"
    saved = read_pickle(path)
    params = _restore_tree_like(nnx.state(agent.policy, nnx.Param), saved, "frozen actor", allow_fallback=False)
    return nnx.graphdef(agent.policy), params


def actor_function(graph, params):
    @jax.jit
    def action(observation):
        return nnx.merge(graph, params).deterministic(observation[None])[0]
    return action


class ExperimentSAC(SACBase):
    """The common SAC, retaining its RNG and protocol when resumed remotely."""

    def __init__(self, obs_dim, act_dim, settings, identity):
        self.identity = identity
        super().__init__(obs_dim, act_dim, settings, seed=settings.seed)

    def checkpoint_state(self):
        leaves = jax.tree.leaves(nnx.state(self.rngs))
        return {"identity": self.identity, "rng_leaves": [
            np.asarray(jax.random.key_data(value)) if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
            else np.asarray(value) for value in leaves]}

    def load_checkpoint_state(self, state):
        if state["identity"] != self.identity:
            raise ValueError("checkpoint belongs to a different experiment")
        leaves, tree = jax.tree.flatten(nnx.state(self.rngs))
        restored = [jax.random.wrap_key_data(jnp.asarray(saved))
                    if jax.dtypes.issubdtype(template.dtype, jax.dtypes.prng_key) else jnp.asarray(saved)
                    for template, saved in zip(leaves, state["rng_leaves"], strict=True)]
        nnx.update(self.rngs, jax.tree.unflatten(tree, restored))


def build_training_scan(world, graph, evidence, conditioned, *, unroll=1, controls=None, temporal=True):
    uniform = jnp.full(4, 0.25)
    reset = world.reset
    simulator_step = world.step
    gains = world.gains
    sys = world.sys
    if conditioned and controls is None:
        _, _, _, controls = frozen_inverse()

    @jax.jit
    def scan(params, state, belief, episode_step, keys, modes):
        def body(carry, inputs):
            current, before, position = carry
            key, mode = inputs
            action_key, noise_key, reset_key = jax.random.split(key, 3)
            observation = current.obs
            augmented = jnp.concatenate((observation, before)) if conditioned else observation
            command, _ = nnx.merge(graph, params).sample(augmented[None], action_key)
            command = command[0]
            executed = apply_action_disturbance(command, noise_key, gains[mode], 0.02)
            following = simulator_step(current, executed)
            after = update_belief(before, evidence(observation, command, following.obs), controls, temporal) if conditioned else before
            terminal = following.done > 0.5
            # HC preserves the existing 4000-step training rollout convention.
            # Reacher resets at the explicitly frozen 200-step time limit.
            timeout = (position + 1 >= world.horizon) if world.kind == "reacher" else False
            restart = terminal | timeout
            outgoing = jax.lax.cond(restart, lambda: reset(sys, reset_key), lambda: following)
            next_observation = jnp.concatenate((following.obs, after)) if conditioned else following.obs
            transition = (augmented, command, following.reward, next_observation, following.done)
            return (outgoing, jnp.where(restart, uniform, after), jnp.where(restart, 0, position + 1)), transition

        return jax.lax.scan(body, (state, belief, episode_step), (keys, modes), unroll=unroll)

    return scan


def training_modes(seed, count, dwell):
    rng = np.random.default_rng(seed + 73001)
    modes = [0]
    for unused in range((count + dwell - 1) // dwell - 1):
        modes.append(int(rng.choice([mode for mode in range(4) if mode != modes[-1]])))
    return np.repeat(np.asarray(modes, np.int32), dwell)[:count]


def train_policy(kind, arm, seed, smoke=False, sampling_device="gpu", *,
                 run_root=None, bundle_root=None, protocol_name=None, conditioning=None):
    if kind == "hc" and arm not in ("reference", "conditioned"):
        raise ValueError("HC comparison has reference and conditioned arms")
    if kind == "reacher" and arm not in ("robust", "reference", "conditioned", "conditioned_no_filter"):
        raise ValueError("unknown Reacher controller arm")
    conditioned = arm in ("conditioned", "conditioned_no_filter")
    if kind == "reacher" and conditioned and conditioning is None:
        raise ValueError("Reacher conditioning requires its frozen estimator and controls")
    destination = p.bundle_dir(kind, arm, seed) if bundle_root is None else bundle_root / kind / arm / f"seed_{seed}"
    if not smoke and (destination / "bundle.json").is_file():
        return
    sampler = jax.devices("cpu")[0] if sampling_device == "cpu" else jax.devices()[0]
    if sampling_device == "cpu":
        if jax.default_backend() != "gpu":
            raise ValueError("CPU sampling still requires a GPU learner")
        if kind != "reacher" or (jax.__version__, version("brax")) != ("0.4.38", "0.10.5"):
            raise ValueError("CPU-sampling protocol requires Reacher, JAX 0.4.38 and Brax 0.10.5")
        if not smoke and not json.loads(p.CPU_ADMISSION.read_text())["passed"]:
            raise ValueError("CPU sampler admission has not passed")
    with jax.default_device(sampler):
        world = World(kind, seed)
    settings = config(kind, seed)
    identity = {"kind": kind, "arm": arm, "seed": seed,
                "protocol": protocol_name or ("mechanism-gap-v34-reacher-cpu" if sampling_device == "cpu" else "mechanism-gap-v34")}
    agent = ExperimentSAC(world.obs_dim + (4 if conditioned else 0), world.act_dim, settings, identity)
    gpu_smoke = smoke and jax.default_backend() == "gpu"
    replay = ReplayBuffer(agent.obs_dim, world.act_dim,
                          capacity=1024 if smoke and not gpu_smoke else settings.replay_size)
    evidence, controls, temporal = None, None, True
    if conditioned:
        if conditioning is None:
            model, params, variance, controls = frozen_inverse()
            evidence = inverse_evidence(model, params, world.gains, variance)
        else:
            evidence, controls, temporal = conditioning
    with jax.default_device(sampler):
        scan = build_training_scan(world, nnx.graphdef(agent.policy), evidence, conditioned,
                                   controls=controls, temporal=temporal)
    directory = p.run_dir(kind, arm, seed) if run_root is None else run_root / kind / arm / f"seed_{seed}"
    logger = Logger(str(directory / "logs"))
    maximum = p.HC_ITERS if kind == "hc" else p.REACHER_ITERS
    start = steps = 0
    checkpoint = directory / "checkpoints"
    if not smoke and (checkpoint / "train_state.pkl").is_file():
        start, steps = load_checkpoint(str(checkpoint), agent, replay, logger, "sac")
        if steps != start * p.SAMPLES or agent.update_count != start * p.UPDATES:
            raise ValueError("resumed iteration, environment steps, and optimizer count disagree")
    count = 16 if smoke else p.SAMPLES
    update_count = 2 if smoke and (not gpu_smoke or kind != "hc") else p.UPDATES
    modes = training_modes(seed, maximum * p.SAMPLES, world.dwell)
    reset = jax.jit(world.reset)
    beginning = time.perf_counter()
    for iteration in range(start, 1 if smoke else maximum):
        iteration_start = time.perf_counter()
        if kind == "hc" and arm == "reference" and iteration == p.FORK_ITER:
            replay.clear()
            agent.policy_opt_state = agent.policy_opt.init(nnx.state(agent.policy, nnx.Param))
            agent.critic_opt_state = agent.critic_opt.init(nnx.state(agent.critic, nnx.Param))
            agent.alpha_opt_state = agent.alpha_opt.init(agent.log_alpha)
            print("FORK mode0: retain networks/alpha, reset replay and optimizers", flush=True)
        with jax.default_device(sampler):
            key = jax.random.fold_in(jax.random.PRNGKey(seed), iteration)
        reset_key, rollout_key, batch_key = jax.random.split(key, 3)
        selected = modes[iteration * p.SAMPLES:iteration * p.SAMPLES + count]
        if arm == "reference" and (kind == "reacher" or iteration >= p.FORK_ITER):
            selected = np.zeros(count, np.int32)
        with jax.default_device(sampler):
            _, transitions = scan(
                jax.device_put(nnx.state(agent.policy, nnx.Param), sampler), reset(world.sys, reset_key),
                jnp.full(4, 0.25), jnp.asarray(0), jax.random.split(rollout_key, count), jnp.asarray(selected))
        learner = jax.devices()[0]
        transitions = jax.device_put(transitions, learner)
        batch_key = jax.device_put(batch_key, learner)
        jax.block_until_ready(transitions)
        rollout_seconds = time.perf_counter() - iteration_start
        update_start = time.perf_counter()
        obs, action, reward, next_obs, done = transitions
        replay.push_batch_jax(obs, action, reward, next_obs, done)
        metrics = agent.multi_update(replay.sample_stacked(update_count, settings.batch_size, batch_key))
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError(f"nonfinite SAC update at iteration {iteration}: {metrics}")
        steps += count
        logger.log_dict({"iteration": iteration, "total_steps": steps,
                         "rollout_update_seconds": time.perf_counter() - iteration_start,
                         "rollout_seconds": rollout_seconds,
                         "update_seconds": time.perf_counter() - update_start, **metrics})
        if sampling_device == "cpu" or iteration - start < 3 or (iteration + 1) % 10 == 0 or iteration + 1 == maximum or smoke:
            print(f"Iter {iteration+1}/{maximum} | steps={steps} | updates={agent.update_count} | "
                  f"Time: {time.perf_counter()-beginning:.1f}s | rollout={rollout_seconds:.3f}s | "
                  f"update={time.perf_counter()-update_start:.3f}s | {metrics}", flush=True)
        if not smoke and ((iteration + 1) % 50 == 0 or iteration + 1 == maximum):
            save_start = time.perf_counter()
            save_checkpoint(str(checkpoint), agent, replay, logger, iteration, steps, "sac")
            logger.save()
            print(f"CHECKPOINT next_iteration={iteration+1} save_seconds={time.perf_counter()-save_start:.3f}", flush=True)
    if smoke:
        return {"kind": kind, "arm": arm, "steps": steps, "updates": agent.update_count, "finite": True}
    if steps != maximum * p.SAMPLES or agent.update_count != maximum * p.UPDATES:
        raise ValueError("final controller budget is incomplete")
    save_pickle(destination / "policy_params.pkl", _to_numpy_tree(nnx.state(agent.policy, nnx.Param)))
    write_json(destination / "bundle.json", {
        **identity, "obs_dim": world.obs_dim, "act_dim": world.act_dim,
        "policy_input_dim": agent.obs_dim, "next_iteration": maximum,
        "total_steps": steps, "update_count": agent.update_count,
        "policy_parameters": sum(value.size for value in jax.tree.leaves(nnx.state(agent.policy, nnx.Param))),
        "elapsed_this_attempt_seconds": time.perf_counter() - beginning,
        "estimator_training_additional": conditioned,
        "belief_controls": list(controls) if conditioned else None,
        "temporal_filter": temporal if conditioned else None,
        "rollout_update_seconds_all_retained_iterations": sum(logger.data["rollout_update_seconds"]),
        "selection": "final", "actor_update_period": 1,
        "sampling_device": sampling_device,
        "runtime_versions": {name: version(name) for name in ("jax", "brax", "flax", "mujoco")},
    })


def load_actor(kind, arm, seed):
    directory = p.bundle_dir(kind, arm, seed)
    manifest = json.loads((directory / "bundle.json").read_text())
    if manifest.get("sampling_device") == "cpu" and any(
        version(name) != saved for name, saved in manifest["runtime_versions"].items()):
        raise ValueError("CPU-sampling evaluation runtime differs from the frozen training bundle")
    expected_iters = p.HC_ITERS if kind == "hc" else p.REACHER_ITERS
    if (manifest["kind"], manifest["arm"], manifest["seed"], manifest["total_steps"], manifest["update_count"]) != (
            kind, arm, seed, expected_iters * p.SAMPLES, expected_iters * p.UPDATES):
        raise ValueError("incorrect/incomplete controller bundle")
    agent = SACBase(manifest["policy_input_dim"], manifest["act_dim"], config(kind, seed), seed)
    params = _restore_tree_like(nnx.state(agent.policy, nnx.Param), read_pickle(directory / "policy_params.pkl"), "new actor", allow_fallback=False)
    return nnx.graphdef(agent.policy), params, manifest


def collect_identification_data(seed, smoke=False):
    destination = p.dataset_dir(seed)
    if not smoke and (destination / "complete.json").is_file():
        return
    world = World("hc", p.PROVIDER_SEED)
    graph, params = load_frozen_actor(p.PROVIDER_SEED)
    gains = world.gains
    beginning = time.perf_counter()

    @partial(jax.jit, static_argnums=1)
    def collect(key, length):
        def body(state, pair):
            step, step_key = pair
            act_key, noise_key, reset_key, explore_key = jax.random.split(step_key, 4)
            mode = (step // 250) % 4
            nominal = nnx.merge(graph, params).deterministic(state.obs[None])[0]
            command = nominal * gains[0] * gains[mode]
            command = jnp.clip(command + 0.15 * jax.random.normal(act_key, command.shape), -1, 1)
            command = jnp.where(jax.random.uniform(explore_key) < 0.2,
                                jax.random.uniform(act_key, command.shape, minval=-1, maxval=1), command)
            executed = apply_action_disturbance(command, noise_key, gains[mode], 0.02)
            following = world.step(state, executed)
            restart = (following.done > 0.5) | ((step + 1) % 1000 == 0)
            outgoing = jax.lax.cond(restart, lambda: world.reset(world.sys, reset_key), lambda: following)
            return outgoing, (state.obs, command, following.obs, mode, command * gains[mode])
        reset_key, rollout_key = jax.random.split(key)
        return jax.lax.scan(body, world.reset(world.sys, reset_key),
                            (jnp.arange(length), jax.random.split(rollout_key, length)))[1]

    for split, count, seed in (("train", p.TRAIN_ROWS, p.TRAIN_EVENT_SEED),
                               ("validation", p.VALIDATION_ROWS, p.VALIDATION_EVENT_SEED)):
        count = 16 if smoke else count
        values = collect(jax.random.PRNGKey(seed), count)
        data = dict(zip(("obs", "action", "next_obs", "mode", "target"), map(np.asarray, values)))
        if not all(np.all(np.isfinite(value)) for value in data.values()):
            raise ValueError("nonfinite identification calibration trajectory")
        if not smoke:
            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination / f"{split}.npz", **data)
        print(f"DATA split={split} rows={count}", flush=True)
    if not smoke:
        write_json(destination / "complete.json", {
            "provider_seed": p.PROVIDER_SEED, "train_rows": p.TRAIN_ROWS,
            "validation_rows": p.VALIDATION_ROWS,
            "source": "new paired calibration trajectories; not reconstructed original V5 data",
            "mode_labels": "training/validation only", "noise_std": 0.02,
            "elapsed_seconds": time.perf_counter() - beginning,
        })


def fit_estimators(seed, smoke=False):
    destination = p.model_dir(seed)
    if not smoke and (destination / "complete.json").is_file():
        return
    if smoke:
        rng = np.random.default_rng(seed)
        obs = rng.normal(size=(512, 17)).astype(np.float32)
        train = {"obs": obs, "next_obs": obs + 0.01, "action": rng.uniform(-1, 1, (512, 6)).astype(np.float32),
                 "target": rng.uniform(-1, 1, (512, 6)).astype(np.float32), "mode": rng.integers(0, 4, 512)}
        validation = train
    else:
        collect_identification_data(seed)
        with np.load(p.dataset_dir(seed) / "train.npz") as archive:
            train = dict(archive)
        with np.load(p.dataset_dir(seed) / "validation.npz") as archive:
            validation = dict(archive)
    models = {"inverse": inverse_model(17, 6, seed), "classifier": DirectClassifier(17, 6, seed)}
    optimizer = optax.chain(optax.clip_by_global_norm(10), optax.adamw(1e-4, weight_decay=1e-5))
    states = {name: (nnx.state(model, nnx.Param), optimizer.init(nnx.state(model, nnx.Param))) for name, model in models.items()}
    graphdefs = {name: nnx.graphdef(model) for name, model in models.items()}
    updates = {}
    for name in models:
        def build(name):
            @jax.jit
            def update(params, opt_state, obs, action, next_obs, target, mode):
                def loss(candidate):
                    model = nnx.merge(graphdefs[name], candidate)
                    if name == "inverse":
                        return jnp.mean(jnp.square(model.predict(obs, next_obs) - target[None]))
                    return optax.softmax_cross_entropy_with_integer_labels(model(obs, action, next_obs), mode).mean()
                value, grads = jax.value_and_grad(loss)(params)
                change, opt_state = optimizer.update(grads, opt_state, params)
                return optax.apply_updates(params, change), opt_state, value
            return update
        updates[name] = build(name)
    checkpoint = p.run_dir("identification", "paired", seed) / "checkpoints/train_state.pkl"
    start = 0
    if not smoke and checkpoint.is_file():
        saved = read_pickle(checkpoint)
        if saved["seed"] != seed:
            raise ValueError("estimator checkpoint seed changed")
        states = _restore_tree_like(states, saved["states"], "estimator training", allow_fallback=False)
        start = saved["next_iteration"]
        print(f"RESUME estimator update={start}", flush=True)
    maximum = 2 if smoke else p.MODEL_UPDATES
    beginning = time.perf_counter()
    for iteration in range(start, maximum):
        rng = np.random.default_rng(np.random.SeedSequence([seed, iteration]))
        indices = rng.integers(len(train["obs"]), size=p.BATCH_SIZE)
        batch = tuple(jnp.asarray(train[key][indices]) for key in ("obs", "action", "next_obs", "target", "mode"))
        for name in models:
            params, opt_state, loss = updates[name](*states[name], *batch)
            if not np.isfinite(float(loss)):
                raise ValueError(f"nonfinite {name} loss at update {iteration}")
            states[name] = params, opt_state
        if (iteration + 1) % 100 == 0 or iteration + 1 == maximum:
            print(f"Iter {iteration+1}/{maximum} | paired estimator updates={iteration+1}/{maximum} | "
                  f"Time: {time.perf_counter()-beginning:.3f}s", flush=True)
            if not smoke:
                save_pickle(checkpoint, {"seed": seed, "iteration": iteration, "next_iteration": iteration + 1,
                                         "total_steps": iteration + 1, "states": _to_numpy_tree(states)})
    if smoke:
        return {"paired_estimator_updates": maximum, "finite": True}
    model = nnx.merge(graphdefs["inverse"], states["inverse"][0])
    predictions = []
    for begin in range(0, len(validation["obs"]), 1024):
        obs = jnp.asarray(validation["obs"][begin:begin+1024])
        next_obs = jnp.asarray(validation["next_obs"][begin:begin+1024])
        predictions.append(np.asarray(model.predict(obs, next_obs)).mean(axis=0))
    errors = np.concatenate(predictions) - validation["target"]
    variance = np.clip(np.mean(errors ** 2, axis=0), 1e-4, 0.25)
    for name in models:
        save_pickle(destination / f"{name}_params.pkl", _to_numpy_tree(states[name][0]))
    write_json(destination / "complete.json", {
        "seed": seed, "train_rows": len(train["obs"]), "validation_rows": len(validation["obs"]),
        "updates_per_model": maximum, "batch_size": p.BATCH_SIZE,
        "same_minibatch_indices": True, "inverse_heads": 5, "classifier_heads": 1,
        "residual_variance": variance.tolist(), "filter_config": {"hazard_rate": 0.002, "evidence_scale": 1.0, "posterior_decay": 0.98},
        "parameters": {name: sum(value.size for value in jax.tree.leaves(states[name][0])) for name in models},
        "elapsed_this_attempt_seconds": time.perf_counter() - beginning,
        "data_collection_seconds": json.loads((p.dataset_dir(seed) / "complete.json").read_text())["elapsed_seconds"],
    })


def learned_evidence(seed, name, gains):
    directory = p.model_dir(seed)
    manifest = json.loads((directory / "complete.json").read_text())
    model = inverse_model(17, 6, seed) if name == "inverse" else DirectClassifier(17, 6, seed)
    params = _restore_tree_like(nnx.state(model, nnx.Param), read_pickle(directory / f"{name}_params.pkl"), name, allow_fallback=False)
    if name == "inverse":
        return inverse_evidence(model, params, gains, jnp.asarray(manifest["residual_variance"]))
    graph = nnx.graphdef(model)

    @jax.jit
    def evidence(observation, action, next_observation):
        logits = nnx.merge(graph, params)(observation[None], action[None], next_observation[None])[0]
        return jax.nn.log_softmax(logits)

    return evidence


def build_episode(world, actor_graph, route, evidence=None, temporal=True):
    gains = world.gains
    controls = (0.002, 1.0, 0.98)
    reset = world.reset
    simulator_step = world.step
    sys = world.sys
    horizon, dwell = world.horizon, world.dwell

    @jax.jit
    def episode(params, key, schedule):
        reset_key, rollout_key = jax.random.split(key)
        initial = reset(sys, reset_key)
        uniform = jnp.full(4, 0.25)

        def body(carry, inputs):
            state, belief, alive = carry
            step, noise_key = inputs
            mode = schedule[(step // dwell) % len(schedule)]
            estimated = jnp.argmax(belief)
            conditioned = route in ("conditioned", "conditioned_no_filter") or route.startswith("conditioned_")
            actor_obs = jnp.concatenate((state.obs, belief)) if conditioned else state.obs
            nominal = nnx.merge(actor_graph, params).deterministic(actor_obs[None])[0]
            if route == "oracle":
                command = compensated_command(nominal, gains, mode)
            elif route.startswith("fixed_"):
                command = compensated_command(nominal, gains, int(route[-1]))
            elif route in ("native_reference", "robust", "uncompensated") or conditioned:
                command = nominal
            else:
                command = compensated_command(nominal, gains, estimated)
            physical_mode = 0 if route == "native_reference" else mode
            executed = apply_action_disturbance(command, noise_key, gains[physical_mode], 0.02)
            following = simulator_step(state, executed)
            after = update_belief(belief, evidence(state.obs, command, following.obs), controls, temporal) if evidence else belief
            following_alive = alive & (following.done < 0.5)
            ideal = nominal_signal(nominal, gains[0])
            mismatch = jnp.sum(jnp.square(nominal_signal(command, gains[physical_mode]) - ideal))
            transition = (jnp.where(alive, following.reward, 0), mode, belief,
                          jnp.where(alive, mismatch, 0), alive, command, following.obs)
            return (following, after, following_alive), transition

        return jax.lax.scan(body, (initial, uniform, jnp.asarray(True)),
                            (jnp.arange(horizon), jax.random.split(rollout_key, horizon)))[1]

    return episode


def trajectory_metrics(values, dwell, route):
    reward, labels, beliefs, energy, alive, commands, observations = map(np.asarray, values)
    if not all(np.all(np.isfinite(value)) for value in (reward, beliefs, energy, commands, observations)):
        raise ValueError("nonfinite frozen-policy evaluation")
    estimated = np.argmax(beliefs, axis=-1)
    valid = alive.astype(bool)
    switched = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    compensation_metric = route not in ("conditioned", "robust") and not route.startswith("conditioned_")
    posterior_metrics = route not in ("robust", "oracle", "native_reference", "uncompensated") and not route.startswith("fixed_")
    delays = []
    for index, start in enumerate(switched):
        end = int(switched[index + 1]) if index + 1 < len(switched) else len(labels)
        detected = next((step - int(start) for step in range(int(start), end - 7)
                         if valid[step:step+8].all() and (estimated[step:step+8] == labels[start]).all()), None)
        delays.append({"switch_step": int(start), "observed": bool(valid[start]), "delay": detected,
                       "window_steps": int(np.sum(valid[start:end]))})
    windows = {}
    for width in (1, 2, 4, 8, 16, 32):
        mask = np.zeros(len(labels), bool)
        for start in switched:
            mask[start:min(start+width, len(labels))] = True
        selected = mask & valid
        windows[str(width)] = {
            "observed_steps": int(selected.sum()),
            "reward_sum": float(reward[selected].sum()),
            "action_error_energy_sum": float(energy[selected].sum()) if compensation_metric else None,
            "mode_error_rate": float(np.mean(estimated[selected] != labels[selected])) if posterior_metrics and selected.any() else None,
        }
    return {"return": float(reward.sum()), "steps": int(valid.sum()), "switches_observed": sum(row["observed"] for row in delays),
            "termination": bool(not valid[-1]), "action_error_energy": float(energy.sum()) if compensation_metric else None,
            "mode_accuracy": float(np.mean(estimated[valid] == labels[valid])) if posterior_metrics else None,
            "brier": float(np.mean(np.sum((beliefs[valid] - np.eye(4)[labels[valid]]) ** 2, axis=-1))) if posterior_metrics else None,
            "delays": delays if posterior_metrics else [], "switch_windows": windows}


def evaluate_routes(world, actors, routes, *, event_seeds=None, episodes=None):
    rows = {}
    trajectories = {}
    event_seeds = p.EVENT_SEEDS if event_seeds is None else event_seeds
    episodes = p.EPISODES if episodes is None else episodes
    for event_index, event_seed in enumerate(event_seeds):
        base = np.roll(np.asarray((0, 1, 3, 2), np.int32), event_index)
        for episode_index in range(episodes):
            schedule = jnp.asarray(np.roll(base, episode_index % 4))
            key = jax.random.PRNGKey(event_seed * 10 + episode_index)
            for route, actor_name, evidence, temporal in routes:
                graph, params = actors[actor_name][:2]
                if route not in trajectories:
                    trajectories[route] = build_episode(world, graph, route, evidence, temporal)
                start = time.perf_counter()
                values = trajectories[route](params, key, schedule)
                jax.block_until_ready(values)
                elapsed = time.perf_counter() - start
                result = trajectory_metrics(values, world.dwell, route)
                result.update(event_seed=event_seed, episode=episode_index,
                              runtime_seconds=elapsed, includes_compilation=event_index == 0 and episode_index == 0)
                rows.setdefault(route, []).append(result)
    return rows


def audit_ablations(seed):
    output = p.audit_path("identification", seed)
    if output.is_file():
        return
    world = World("hc", seed)
    actors = {"reference": load_frozen_actor(seed)}
    model, params, variance, _ = frozen_inverse()
    full = inverse_evidence(model, params, world.gains, variance)
    single = inverse_evidence(model, params, world.gains, variance, single_head=True)
    routes = [(name, "reference", full if name not in ("oracle", "uncompensated") else None, name != "no_filter") for name in p.FROZEN_ROUTES]
    routes[2] = ("single_head", "reference", single, True)
    for model_seed in p.MODEL_SEEDS:
        for name in ("inverse", "classifier"):
            routes.append((f"{name}_{model_seed}", "reference", learned_evidence(model_seed, name, world.gains), True))
    events = evaluate_routes(world, actors, routes)
    write_json(output, {"kind": "identification", "seed": seed, "events": events,
                        "means": {name: float(np.mean([row["return"] for row in values])) for name, values in events.items()},
                        "provider_seed_excluded": p.PROVIDER_SEED})


def audit_policy(kind, seed):
    output = p.audit_path(kind, seed)
    if output.is_file():
        return
    world = World(kind, seed)
    arms = ("reference", "conditioned") if kind == "hc" else ("reference", "robust")
    actors = {arm: load_actor(kind, arm, seed) for arm in arms}
    evidence = None
    if kind == "hc":
        model, params, variance, _ = frozen_inverse()
        evidence = inverse_evidence(model, params, world.gains, variance)
    routes = [("oracle", "reference", None, True), ("native_reference", "reference", None, True),
              ("uncompensated", "reference", None, True)]
    if kind == "hc":
        routes.extend((("frozen", "reference", evidence, True), ("conditioned", "conditioned", evidence, True)))
    else:
        routes.append(("robust", "robust", None, True))
        routes.extend((f"fixed_{mode}", "reference", None, True) for mode in range(4))
    events = evaluate_routes(world, actors, routes)
    paired = np.asarray([row["return"] for row in events["oracle"]]) - np.asarray([row["return"] for row in events["native_reference"]])
    if not np.allclose(paired, 0, atol=1e-3):
        raise ValueError("oracle/native paired trajectory equivalence failed")
    write_json(output, {"kind": kind, "seed": seed, "events": events,
                        "means": {name: float(np.mean([row["return"] for row in values])) for name, values in events.items()},
                        "budgets": {arm: values[2] for arm, values in actors.items()},
                        "oracle_native_max_abs_return_difference": float(np.max(np.abs(paired)))})


def paired_summary(left, right):
    difference = np.asarray(left, np.float64) - np.asarray(right, np.float64)
    radius = student_t.ppf(0.975, len(difference) - 1) * np.std(difference, ddof=1) / np.sqrt(len(difference))
    return {"n_policy_seeds": len(difference), "mean": float(np.mean(difference)),
            "ci95": [float(np.mean(difference) - radius), float(np.mean(difference) + radius)],
            "wins": int(np.sum(difference > 0)), "differences": difference.tolist()}


def analyze(kind):
    seeds = p.ABLATION_SEEDS if kind == "identification" else p.POLICY_SEEDS if kind == "hc" else p.REACHER_SEEDS
    payloads = [json.loads(p.audit_path(kind, seed).read_text()) for seed in seeds]
    arms = tuple(payloads[0]["means"])
    means = {arm: [row["means"][arm] for row in payloads] for arm in arms}
    comparisons = {}
    if kind == "identification":
        for arm in arms:
            if arm != "frozen":
                comparisons[f"frozen-minus-{arm}"] = paired_summary(means["frozen"], means[arm])
        for model_seed in p.MODEL_SEEDS:
            comparisons[f"paired-inverse-minus-classifier-{model_seed}"] = paired_summary(means[f"inverse_{model_seed}"], means[f"classifier_{model_seed}"])
    elif kind == "hc":
        comparisons["compensation-minus-information-matched-SAC"] = paired_summary(means["frozen"], means["conditioned"])
    else:
        for arm in ("robust", *[f"fixed_{mode}" for mode in range(4)]):
            comparisons[f"oracle-minus-{arm}"] = paired_summary(means["oracle"], means[arm])
    payload = {"kind": kind, "seeds": list(seeds), "seed_means": means,
               "means": {arm: float(np.mean(values)) for arm, values in means.items()},
               "comparisons": comparisons, "inference_unit": "policy seed, not episode"}
    if kind == "reacher":
        headroom = comparisons["oracle-minus-robust"]
        all_full = all(value["steps"] == p.REACHER_HORIZON and value["switches_observed"] == 3
                       for row in payloads for name in ("robust", "oracle") for value in row["events"][name])
        minimum_gain = 0.1 * max(1.0, abs(payload["means"]["robust"]))
        payload["headroom_screen"] = {"valid_windows": all_full,
            "minimum_absolute_gain": minimum_gain,
            "passes": bool(all_full and headroom["wins"] == len(seeds) and headroom["mean"] > minimum_gain
                           and all(value["mean"] > 0 for value in comparisons.values())),
            "next": "train a Reacher inverse estimator only after positive headroom; otherwise retain a negative applicability result"}
    write_json(p.analysis_path(kind), payload)
    print(json.dumps(payload, indent=2), flush=True)


def smoke():
    if jax.default_backend() != "gpu":
        raise ValueError("The production smoke gate must measure a remote GPU")
    # Functional smoke uses small replay; allocation-only check measures the
    # actual production replay before any dependent job can start.
    production_replay = ReplayBuffer(21, 6, capacity=1000000)
    jax.block_until_ready(production_replay.obs)
    allocation = jax.devices()[0].memory_stats() or {}
    del production_replay
    rows = [train_policy("hc", "conditioned", 95001, smoke=True),
            train_policy("reacher", "reference", 95003, smoke=True), fit_estimators(95005, smoke=True)]
    world = World("reacher", 95003)
    agent = SACBase(world.obs_dim, world.act_dim, config("reacher", 95003), 95003)
    graph, params = nnx.graphdef(agent.policy), nnx.state(agent.policy, nnx.Param)
    key, schedule = jax.random.PRNGKey(95007), jnp.asarray((0, 1, 3, 2))
    oracle = build_episode(world, graph, "oracle")(params, key, schedule)
    native = build_episode(world, graph, "native_reference")(params, key, schedule)
    for position in (0, 6):
        if not np.allclose(np.asarray(oracle[position]), np.asarray(native[position]), atol=1e-4):
            raise ValueError("Reacher oracle/native trajectory mismatch in smoke")
    collect_identification_data(95005, smoke=True)
    stats = jax.devices()[0].memory_stats() or {}
    write_json(p.BUNDLE_ROOT / "smoke/ready.json", {"finite": True, "runs": rows,
               "device": str(jax.devices()[0]), "jax_version": jax.__version__,
               "allocation_stats": allocation, "memory_stats": stats,
               "production_replay_capacity": 1000000})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train", "collect", "fit", "ablate", "audit", "analyze"))
    parser.add_argument("--kind", choices=("hc", "reacher", "identification"), default="hc")
    parser.add_argument("--arm", choices=("reference", "conditioned", "robust"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sampling-device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    if args.sampling_device == "cpu":
        if args.kind != "reacher" or args.mode not in ("train", "audit", "analyze"):
            parser.error("CPU sampling selects the separate Reacher train/audit/analyze protocol")
        p.RUN_ROOT, p.BUNDLE_ROOT = p.CPU_RUN_ROOT, p.CPU_BUNDLE_ROOT
    if args.mode == "smoke": smoke()
    elif args.mode == "train": train_policy(args.kind, args.arm, args.seed, sampling_device=args.sampling_device)
    elif args.mode == "collect": collect_identification_data(args.seed)
    elif args.mode == "fit": fit_estimators(args.seed)
    elif args.mode == "ablate": audit_ablations(args.seed)
    elif args.mode == "audit": audit_policy(args.kind, args.seed)
    elif args.mode == "analyze": analyze(args.kind)


if __name__ == "__main__":
    main()
