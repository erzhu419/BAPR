"""Brax GPU-accelerated Non-stationary Environments.

V4: Everything fused in lax.scan — zero Python loop overhead.
- step_fn with explicit sys arg → compiled ONCE
- rollout_with_policy: fuses policy(obs)→action→physics→auto-reset in lax.scan
- sys is passed as pytree arrays → JAX traces once, reuses for all tasks
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict
from brax import envs
from brax.envs.base import State


RAND_PARAMS_MAP = {
    # Map MuJoCo-style logical names to the Brax System fields that are actually
    # read by the spring/generalized pipelines. Top-level body_mass,
    # body_inertia, and dof_damping exist as legacy copies, but replacing them is
    # a silent no-op for physics.
    'gravity': 'gravity',
    'body_mass': 'link.inertia.mass',
    'dof_damping': 'dof.damping',
    'body_inertia': 'link.inertia.i',
}

BRAX_ENV_MAP = {
    'Hopper-v2': 'hopper', 'Hopper-v4': 'hopper', 'Hopper-v5': 'hopper',
    'HalfCheetah-v2': 'halfcheetah', 'HalfCheetah-v4': 'halfcheetah', 'HalfCheetah-v5': 'halfcheetah',
    'Walker2d-v2': 'walker2d', 'Walker2d-v4': 'walker2d', 'Walker2d-v5': 'walker2d',
    'Ant-v2': 'ant', 'Ant-v4': 'ant', 'Ant-v5': 'ant',
}


def apply_action_disturbance(
        action, rng, gain, noise_std, packet_loss_prob=0.0,
        burst_prob=0.0, burst_std=0.0):
    """Apply mode-conditioned actuator dynamics before the physics step.

    The replay action remains the policy command.  A packet-loss event drops
    the whole command for one simulator step, while a burst event adds a
    heavy-tailed torque impulse.  Their probabilities are fixed within a
    persistent mode but sampled independently on each transition.
    """
    action = jnp.asarray(action)
    gain = jnp.asarray(gain, dtype=action.dtype)
    noise_std = jnp.asarray(noise_std, dtype=action.dtype)
    packet_loss_prob = jnp.clip(
        jnp.asarray(packet_loss_prob, dtype=action.dtype), 0.0, 1.0)
    burst_prob = jnp.clip(
        jnp.asarray(burst_prob, dtype=action.dtype), 0.0, 1.0)
    burst_std = jnp.maximum(
        jnp.asarray(burst_std, dtype=action.dtype), 0.0)
    noise_key, packet_key, burst_key, impulse_key = jax.random.split(rng, 4)
    noise = jax.random.normal(noise_key, action.shape, dtype=action.dtype)
    packet_kept = jnp.logical_not(jax.random.bernoulli(
        packet_key, packet_loss_prob))
    burst_active = jax.random.bernoulli(burst_key, burst_prob)
    impulse = jax.random.normal(
        impulse_key, action.shape, dtype=action.dtype)
    executed = (
        gain * action * packet_kept
        + noise_std * noise
        + burst_active * burst_std * impulse)
    return jnp.clip(executed, -1.0, 1.0)


def _build_core_fns(env, env_name):
    """Build pure functions for env physics, reward, obs, reset."""
    pipeline = env._pipeline
    n_frames = env._n_frames
    debug = env._debug
    dt = env.dt
    brax_name = BRAX_ENV_MAP.get(env_name, env_name.lower())

    fwd_w = getattr(env, '_forward_reward_weight', 1.0)
    ctrl_w = getattr(env, '_ctrl_cost_weight', 1e-3)
    healthy_r = getattr(env, '_healthy_reward', 1.0)
    terminate = getattr(env, '_terminate_when_unhealthy', True)
    hz_range = getattr(env, '_healthy_z_range', (0.7, float('inf')))
    ha_range = getattr(env, '_healthy_angle_range', (-0.2, 0.2))
    hs_range = getattr(env, '_healthy_state_range', (-100.0, 100.0))
    exclude_pos = getattr(env, '_exclude_current_positions_from_observation', True)
    rns = getattr(env, '_reset_noise_scale', 5e-3)

    def physics_step(sys, ps, action):
        def f(s, _):
            return pipeline.step(sys, s, action, debug), None
        return jax.lax.scan(f, ps, (), n_frames)[0]

    def reward_obs(ps0, ps1, action):
        x_vel = (ps1.x.pos[0, 0] - ps0.x.pos[0, 0]) / dt
        if brax_name == 'halfcheetah':
            reward = fwd_w * x_vel - ctrl_w * jnp.sum(jnp.square(action))
            done = jnp.float32(0.0)
            pos = ps1.q[1:] if exclude_pos else ps1.q
            obs = jnp.concatenate([pos, ps1.qd])
        elif brax_name in ('hopper', 'walker2d'):
            z = ps1.x.pos[0, 2]
            angle = ps1.q[2]
            if brax_name == 'hopper':
                sv = jnp.concatenate([ps1.q[2:], ps1.qd])
                is_h = jnp.all(jnp.logical_and(hs_range[0] < sv, sv < hs_range[1]))
                is_h = is_h & (hz_range[0] < z) & (z < hz_range[1])
                is_h = is_h & (ha_range[0] < angle) & (angle < ha_range[1])
            else:
                is_h = ((z > hz_range[0]) & (z < hz_range[1]) &
                        (angle > ha_range[0]) & (angle < ha_range[1]))
            h_rew = jnp.where(terminate, healthy_r, healthy_r * is_h)
            reward = fwd_w * x_vel + h_rew - ctrl_w * jnp.sum(jnp.square(action))
            done = jnp.where(terminate, 1.0 - is_h, 0.0)
            pos = ps1.q.at[1].set(z)
            vel = jnp.clip(ps1.qd, -10, 10)
            if exclude_pos: pos = pos[1:]
            obs = jnp.concatenate([pos, vel])
        elif brax_name == 'ant':
            vel = (ps1.x.pos[0] - ps0.x.pos[0]) / dt
            z = ps1.x.pos[0, 2]
            is_h = jnp.where(z < hz_range[0], 0.0, 1.0)
            is_h = jnp.where(z > hz_range[1], 0.0, is_h)
            h_rew = jnp.where(terminate, healthy_r, healthy_r * is_h)
            reward = vel[0] + h_rew - ctrl_w * jnp.sum(jnp.square(action))
            done = jnp.where(terminate, 1.0 - is_h, 0.0)
            qpos = ps1.q[2:] if exclude_pos else ps1.q
            obs = jnp.concatenate([qpos, ps1.qd])
        else:
            raise ValueError(f"Unsupported: {brax_name}")
        return reward, obs, done

    def init_obs(ps):
        if brax_name == 'halfcheetah':
            pos = ps.q[1:] if exclude_pos else ps.q
            return jnp.concatenate([pos, ps.qd])
        elif brax_name in ('hopper', 'walker2d'):
            pos = ps.q.at[1].set(ps.x.pos[0, 2])
            vel = jnp.clip(ps.qd, -10, 10)
            if exclude_pos: pos = pos[1:]
            return jnp.concatenate([pos, vel])
        elif brax_name == 'ant':
            qpos = ps.q[2:] if exclude_pos else ps.q
            return jnp.concatenate([qpos, ps.qd])
        return jnp.concatenate([ps.q, ps.qd])

    def reset_state(sys, rng):
        r1, r2 = jax.random.split(rng)
        q = sys.init_q + jax.random.uniform(r1, (sys.q_size(),), minval=-rns, maxval=rns)
        qd = jax.random.uniform(r2, (sys.qd_size(),), minval=-rns, maxval=rns)
        ps = pipeline.init(sys, q, qd, debug=debug)
        return State(pipeline_state=ps, obs=init_obs(ps),
                     reward=jnp.float32(0.0), done=jnp.float32(0.0),
                     metrics={}, info={})

    return physics_step, reward_obs, reset_state


class BraxNonstationaryEnv:
    """GPU-accelerated non-stationary env.

    The key optimization: build a SINGLE lax.scan rollout function that fuses
    policy inference + physics + auto-reset. Compiled ONCE, reused for all tasks
    because sys is passed as a normal pytree argument.
    """

    def __init__(self, env_name: str, rand_params: List[str] = None,
                 log_scale_limit: float = 3.0, seed: int = 0,
                 backend: str = 'spring',
                 task_scale_distribution: str = 'exp',
                 task_seed_salt: int = 0):
        brax_name = BRAX_ENV_MAP.get(env_name, env_name.lower())
        self.env = envs.get_environment(brax_name, backend=backend)
        self.base_sys = self.env.sys
        self.env_name = env_name
        self.backend = backend

        self.rand_params = rand_params or ['gravity']
        self.log_scale_limit = log_scale_limit
        if task_scale_distribution not in ('exp', 'pow1p5'):
            raise ValueError(
                "task_scale_distribution must be 'exp' or 'pow1p5', got "
                f"{task_scale_distribution!r}")
        self.task_scale_distribution = task_scale_distribution
        self.seed = seed
        self.task_seed_salt = int(task_seed_salt)
        self.rng = jax.random.PRNGKey(seed)

        self.obs_dim = self.env.observation_size
        self.act_dim = self.env.action_size
        self.dt = self.env.dt

        self._base_values = {}
        for param in self.rand_params:
            key = RAND_PARAMS_MAP.get(param, param)
            if '.' in key:
                val = self.base_sys
                for p in key.split('.'): val = getattr(val, p)
            else:
                val = getattr(self.base_sys, key)
            self._base_values[param] = jnp.array(val)

        self.current_task_id = 0
        self._tasks = None
        self._task_sys_list = None
        self._task_sample_calls = 0
        self._changing_interval = 10
        self._changing_period = 100
        self._step_counter = 0
        self._current_sys = self.base_sys
        self._state = None
        self._action_gain = jnp.ones((self.act_dim,), dtype=jnp.float32)
        self._action_noise_std = jnp.zeros(
            (self.act_dim,), dtype=jnp.float32)
        self._action_packet_loss_prob = jnp.asarray(0.0, dtype=jnp.float32)
        self._action_burst_prob = jnp.asarray(0.0, dtype=jnp.float32)
        self._action_burst_std = jnp.asarray(0.0, dtype=jnp.float32)
        # Nonzero values ask collect_samples to preserve the environment's
        # mode clock by splitting every algorithm's fused rollout equally.
        self.rollout_chunk_steps = 0

        # Build core physics functions
        self._physics_step, self._reward_obs, self._reset_state = \
            _build_core_fns(self.env, env_name)

        # Build JIT'd step/reset (for sequential API)
        physics_step = self._physics_step
        reward_obs = self._reward_obs
        reset_state = self._reset_state

        @jax.jit
        def _step(sys, state, action):
            ps0 = state.pipeline_state
            ps1 = physics_step(sys, ps0, action)
            r, o, d = reward_obs(ps0, ps1, action)
            return state.replace(pipeline_state=ps1, obs=o, reward=r, done=d)

        @jax.jit
        def _reset(sys, rng):
            return reset_state(sys, rng)

        self._step_fn = _step
        self._reset_fn = _reset

    # --- Task management ---

    def _set_sys(self, sys):
        self._current_sys = sys

    def _set_action_disturbance(
            self, gain=1.0, noise_std=0.0, packet_loss_prob=0.0,
            burst_prob=0.0, burst_std=0.0):
        self._action_gain = jnp.broadcast_to(
            jnp.asarray(gain, dtype=jnp.float32), (self.act_dim,))
        self._action_noise_std = jnp.broadcast_to(
            jnp.asarray(noise_std, dtype=jnp.float32), (self.act_dim,))
        self._action_packet_loss_prob = jnp.clip(
            jnp.asarray(packet_loss_prob, dtype=jnp.float32), 0.0, 1.0)
        self._action_burst_prob = jnp.clip(
            jnp.asarray(burst_prob, dtype=jnp.float32), 0.0, 1.0)
        self._action_burst_std = jnp.maximum(
            jnp.asarray(burst_std, dtype=jnp.float32), 0.0)

    def action_disturbance_params(self):
        return (
            self._action_gain,
            self._action_noise_std,
            self._action_packet_loss_prob,
            self._action_burst_prob,
            self._action_burst_std,
        )

    def sample_tasks(self, n_tasks: int) -> List[Dict]:
        """Sample piecewise-stationary tasks with continuous uniform log_scale.

        Matches original archived BAPR design (HC=15702 reference): 40 tasks
        with log_scale uniform in [-limit, +limit]. Most tasks fall near the
        center (mild perturbation) with tails reaching extreme values. BOCD
        activates occasionally on extreme tasks; between them, λ_w ≈ 0 and
        policy learning proceeds normally.

        Discrete extreme sampling (prev version) was too harsh — every task
        was extreme, BOCD constantly fired, β_eff stayed over-conservative.
        """
        tasks = []
        call_id = self._task_sample_calls
        self._task_sample_calls += 1
        # Use deterministic but distinct streams for sequential train/test
        # sampling calls. Previously both calls used self.seed + 42, so
        # train_tasks and test_tasks were identical and OOD eval was mislabeled.
        rng = np.random.RandomState(
            self.seed + self.task_seed_salt + 42 + 100_003 * call_id)
        for _ in range(n_tasks):
            task = {}
            for param in self.rand_params:
                base = np.array(self._base_values[param])
                log_scale = rng.uniform(-self.log_scale_limit, self.log_scale_limit,
                                       size=base.shape)
                if self.task_scale_distribution == 'pow1p5':
                    scale = np.power(1.5, log_scale)
                else:
                    scale = np.exp(log_scale)
                task[param] = base * scale.astype(np.float32)
            tasks.append(task)
        return tasks

    def set_task(self, task: Dict):
        replacements = {}
        for param, value in task.items():
            key = RAND_PARAMS_MAP.get(param, param)
            replacements[key] = jnp.array(value)
        self._set_sys(self.base_sys.tree_replace(replacements))

    def set_nonstationary_para(self, tasks, changing_period=100, changing_interval=10):
        self._tasks = tasks
        self._changing_period = changing_period
        self._changing_interval = changing_interval
        self._step_counter = 0
        self._task_sys_list = []
        for task in tasks:
            replacements = {}
            for param, value in task.items():
                key = RAND_PARAMS_MAP.get(param, param)
                replacements[key] = jnp.array(value)
            self._task_sys_list.append(self.base_sys.tree_replace(replacements))
        self.current_task_id = 0
        self._set_sys(self._task_sys_list[0])

    def _check_switch(self):
        if (self._tasks is not None and
                self._step_counter % self._changing_interval == 0):
            idx = int(self._step_counter / self._changing_period) % len(self._tasks)
            if idx != self.current_task_id:
                self.current_task_id = idx
                self._set_sys(self._task_sys_list[idx])

    def task_id_for_next_step(self) -> int:
        """Return the physics task that the next action will encounter."""
        next_step = self._step_counter + 1
        if (self._tasks is not None and self._tasks
                and next_step % self._changing_interval == 0):
            return int(next_step / self._changing_period) % len(self._tasks)
        return int(self.current_task_id)

    # --- Sequential API ---

    def reset(self):
        self.rng, key = jax.random.split(self.rng)
        self._state = self._reset_fn(self._current_sys, key)
        return np.array(self._state.obs)

    def step(self, action):
        self._step_counter += 1
        self._check_switch()
        self.rng, noise_key = jax.random.split(self.rng)
        action_jax = jnp.array(action)
        executed_action = apply_action_disturbance(
            action_jax, noise_key, self._action_gain,
            self._action_noise_std, self._action_packet_loss_prob,
            self._action_burst_prob, self._action_burst_std)
        self._state = self._step_fn(
            self._current_sys, self._state, executed_action)
        return np.array(self._state.obs), float(self._state.reward), \
               bool(self._state.done), {
                   "executed_action": np.asarray(executed_action),
                   "action_gain": np.asarray(self._action_gain),
                   "action_noise_std": np.asarray(self._action_noise_std),
                   "packet_loss_prob": float(
                       self._action_packet_loss_prob),
                   "burst_prob": float(self._action_burst_prob),
                   "burst_std": float(self._action_burst_std),
               }

    def close(self):
        pass

    @property
    def action_space(self):
        class _AS:
            def __init__(s, dim): s.shape = (dim,)
            def sample(s): return np.random.uniform(-1, 1, size=s.shape).astype(np.float32)
        return _AS(self.act_dim)

    # --- Scan-fused rollout with policy ---

    def build_rollout_fn(self, policy_graphdef, context_graphdef=None,
                         transition_context_graphdef=None,
                         recurrent_context_graphdef=None,
                         recurrent_context_non_params=None,
                         critic_graphdef=None,
                         direct_policy_context=False):
        """Build JIT'd scan rollouts: stochastic (training) + deterministic (eval).

        Args:
            policy_graphdef: nnx.graphdef(agent.policy)
            context_graphdef: optional nnx.graphdef(agent.context_net) for ESCP/BAPR
            transition_context_graphdef: optional causal context encoder for
                BAPR-v2. It consumes transitions inside the same rollout scan.
            recurrent_context_graphdef: optional ESCP recurrent probe. It
                consumes ``(observation, previous_action)`` and carries a GRU
                state causally through the rollout.
            recurrent_context_non_params: exhaustive non-parameter NNX state
                for the recurrent probe. Flax 0.10 stores GRU RNG state here.
            critic_graphdef: optional BAPR-v2 critic used for conservative
                residual-advantage fallback during rollout and evaluation.
            direct_policy_context: feed belief_vec directly to a conditioned
                policy without a state or transition encoder.
        """
        from flax import nnx

        physics_step = self._physics_step
        reward_obs = self._reward_obs
        reset_state = self._reset_state
        has_context = context_graphdef is not None
        has_direct_context = bool(direct_policy_context)
        has_transition_context = transition_context_graphdef is not None
        has_recurrent_context = recurrent_context_graphdef is not None
        if sum(map(int, (
                has_context, has_direct_context, has_transition_context,
                has_recurrent_context))) > 1:
            raise ValueError("rollout context mechanisms are mutually exclusive")
        has_advantage_critic = critic_graphdef is not None
        if has_advantage_critic:
            from jax_experiments.networks.residual_policy import (
                advantage_gated_action,
            )

        def adaptive_context(context_model, adapt, oracle_latent, source):
            learned = context_model.policy_context(adapt, oracle_latent)
            oracle = jnp.concatenate([
                oracle_latent[:context_model.latent_dim],
                jnp.ones((1,), dtype=oracle_latent.dtype),
            ])
            selected = jnp.where(
                source == 0,
                jnp.zeros_like(learned),
                jnp.where(source == 1, oracle, learned),
            )
            gate = jnp.clip(selected[-1:], 0.0, 1.0)
            return jnp.concatenate([selected[:-1] * gate, gate])

        # --- Stochastic rollout (training) ---
        # v15: optional belief_vec arg lets BAPR feed the BOCD posterior into
        # the policy alongside context. None → ESCP/RESAC/SAC unchanged path.
        @jax.jit
        def _rollout_scan(
                sys, action_gain, action_noise_std, packet_loss_prob,
                burst_prob, burst_std, policy_params, context_params,
                belief_vec, init_state, keys, warmup):
            """warmup: jax bool — when True, zero out the context embedding
            so rollout matches the training-time warmup phase (GPT-5.5
            advice #3). belief_vec is also zeroed during warmup.
            """
            def scan_body(carry, key):
                state = carry
                policy_key, noise_key, reset_key = jax.random.split(key, 3)

                pre_obs = state.obs
                policy = nnx.merge(policy_graphdef, policy_params)

                if has_context:
                    ctx_net = nnx.merge(context_graphdef, context_params)
                    ep = ctx_net(pre_obs[None])
                    ep = jnp.where(warmup, jnp.zeros_like(ep), ep)
                    if belief_vec is not None:
                        b = jnp.where(warmup, jnp.zeros_like(belief_vec),
                                       belief_vec)
                        ep = jnp.concatenate([ep, b[None, :]], axis=-1)
                    action, _ = policy.sample(pre_obs[None], policy_key, ep)
                elif has_direct_context:
                    ep = jnp.where(
                        warmup, jnp.zeros_like(belief_vec), belief_vec)
                    action, _ = policy.sample(
                        pre_obs[None], policy_key, ep[None, :])
                else:
                    action, _ = policy.sample(pre_obs[None], policy_key)
                action = action[0]
                executed_action = apply_action_disturbance(
                    action, noise_key, action_gain, action_noise_std,
                    packet_loss_prob, burst_prob, burst_std)

                ps0 = state.pipeline_state
                ps1 = physics_step(sys, ps0, executed_action)
                reward, post_obs, done = reward_obs(
                    ps0, ps1, executed_action)
                next_state = state.replace(
                    pipeline_state=ps1, obs=post_obs, reward=reward, done=done)

                reset_st = reset_state(sys, reset_key)
                out_state = jax.tree.map(
                    lambda r, n: jnp.where(done, r, n), reset_st, next_state)

                transition = (
                    pre_obs, action, reward, post_obs, done, executed_action)
                return out_state, transition

            final_state, transitions = jax.lax.scan(
                scan_body, init_state, keys)
            return final_state, transitions

        @jax.jit
        def _rollout_scan_recurrent(
                sys, action_gain, action_noise_std, packet_loss_prob,
                burst_prob, burst_std, policy_params, context_params,
                init_hidden, init_previous_action, init_state, keys, warmup,
                context_noise_sigma):
            """ESCP rollout with a causal recurrent environment probe."""
            policy = nnx.merge(policy_graphdef, policy_params)
            context_model = nnx.merge(
                recurrent_context_graphdef, context_params,
                recurrent_context_non_params)

            def scan_body(carry, key):
                state, hidden, previous_action = carry
                (policy_key, context_noise_key, process_noise_key,
                 reset_key) = jax.random.split(key, 4)
                pre_obs = state.obs
                next_hidden, context = context_model.step(
                    hidden, pre_obs[None], previous_action[None])
                context = context[0]
                context = jnp.where(
                    warmup, jnp.zeros_like(context), context)
                context = context + jnp.where(
                    warmup, 0.0, context_noise_sigma) * jax.random.normal(
                        context_noise_key, context.shape)
                action, _ = policy.sample(
                    pre_obs[None], policy_key, context[None])
                action = action[0]
                executed_action = apply_action_disturbance(
                    action, process_noise_key, action_gain,
                    action_noise_std, packet_loss_prob, burst_prob,
                    burst_std)

                ps0 = state.pipeline_state
                ps1 = physics_step(sys, ps0, executed_action)
                reward, post_obs, done = reward_obs(
                    ps0, ps1, executed_action)
                next_state = state.replace(
                    pipeline_state=ps1, obs=post_obs,
                    reward=reward, done=done)
                reset_st = reset_state(sys, reset_key)
                out_state = jax.tree.map(
                    lambda reset, current: jnp.where(
                        done, reset, current),
                    reset_st, next_state)
                out_hidden = jnp.where(
                    done, jnp.zeros_like(next_hidden), next_hidden)
                out_previous_action = jnp.where(
                    done, jnp.zeros_like(action), action)
                transition = (
                    pre_obs, action, reward, post_obs, done,
                    executed_action)
                return (
                    out_state, out_hidden, out_previous_action), transition

            final, transitions = jax.lax.scan(
                scan_body,
                (init_state, init_hidden, init_previous_action), keys)
            return final, transitions

        @jax.jit
        def _rollout_scan_adaptive(
                sys, action_gain, action_noise_std, packet_loss_prob,
                burst_prob, burst_std, policy_params, critic_params,
                context_params,
                adaptation_state, oracle_latent, init_state, keys, warmup,
                context_source, advantage_enabled, advantage_margin,
                advantage_lcb_scale):
            """BAPR-v2 rollout with causal context carried through the scan."""
            def scan_body(carry, key):
                state, adapt = carry
                action_key, noise_key, reset_key = jax.random.split(key, 3)
                policy = nnx.merge(policy_graphdef, policy_params)
                context_model = nnx.merge(
                    transition_context_graphdef, context_params)

                pre_obs = state.obs
                context = adaptive_context(
                    context_model, adapt, oracle_latent, context_source)
                policy_context = jnp.where(
                    warmup, jnp.zeros_like(context), context)
                if has_advantage_critic:
                    critic = nnx.merge(critic_graphdef, critic_params)
                    action, advantage, advantage_gate = (
                        advantage_gated_action(
                            policy, critic, pre_obs[None],
                            policy_context[None], key=action_key,
                            enabled=advantage_enabled,
                            margin=advantage_margin,
                            lcb_scale=advantage_lcb_scale))
                    action = action[0]
                    advantage = advantage[0]
                    advantage_gate = advantage_gate[0]
                else:
                    action, _ = policy.sample(
                        pre_obs[None], action_key, policy_context[None])
                    action = action[0]
                    advantage = jnp.asarray(0.0, dtype=action.dtype)
                    advantage_gate = jnp.asarray(1.0, dtype=action.dtype)
                executed_action = apply_action_disturbance(
                    action, noise_key, action_gain, action_noise_std,
                    packet_loss_prob, burst_prob, burst_std)

                ps0 = state.pipeline_state
                ps1 = physics_step(sys, ps0, executed_action)
                reward, post_obs, done = reward_obs(
                    ps0, ps1, executed_action)
                next_state = state.replace(
                    pipeline_state=ps1, obs=post_obs,
                    reward=reward, done=done)

                next_adapt, error, _, _ = context_model.observe(
                    adapt, pre_obs, action, reward, post_obs, done,
                    enable_reset=True)
                next_context = adaptive_context(
                    context_model, next_adapt, oracle_latent, context_source)
                next_policy_context = jnp.where(
                    warmup, jnp.zeros_like(next_context), next_context)

                reset_st = reset_state(sys, reset_key)
                out_state = jax.tree.map(
                    lambda r, n: jnp.where(done, r, n),
                    reset_st, next_state)
                transition = (
                    pre_obs, action, reward, post_obs, done,
                    policy_context, next_policy_context, error,
                    advantage, advantage_gate)
                return (out_state, next_adapt), transition

            (final_state, final_adapt), transitions = jax.lax.scan(
                scan_body, (init_state, adaptation_state), keys)
            return final_state, final_adapt, transitions

        # --- Mean-policy rollout (eval); transition noise remains stochastic ---
        @jax.jit
        def _rollout_scan_det(
                sys, action_gain, action_noise_std, packet_loss_prob,
                burst_prob, burst_std, policy_params, context_params,
                belief_vec, init_state, step_keys, warmup):
            """Eval rollout — uses policy mean (no exploration noise).

            step_keys: per-step keys for process noise and auto-reset sampling.
            warmup: jax bool — same semantics as _rollout_scan.
            """
            def scan_body(carry, step_key):
                state = carry
                noise_key, reset_key = jax.random.split(step_key)
                pre_obs = state.obs
                policy = nnx.merge(policy_graphdef, policy_params)

                if has_context:
                    ctx_net = nnx.merge(context_graphdef, context_params)
                    ep = ctx_net(pre_obs[None])
                    ep = jnp.where(warmup, jnp.zeros_like(ep), ep)
                    if belief_vec is not None:
                        b = jnp.where(warmup, jnp.zeros_like(belief_vec),
                                       belief_vec)
                        ep = jnp.concatenate([ep, b[None, :]], axis=-1)
                    action = policy.deterministic(pre_obs[None], ep)
                elif has_direct_context:
                    ep = jnp.where(
                        warmup, jnp.zeros_like(belief_vec), belief_vec)
                    action = policy.deterministic(
                        pre_obs[None], ep[None, :])
                else:
                    action = policy.deterministic(pre_obs[None])
                action = action[0]
                executed_action = apply_action_disturbance(
                    action, noise_key, action_gain, action_noise_std,
                    packet_loss_prob, burst_prob, burst_std)

                ps0 = state.pipeline_state
                ps1 = physics_step(sys, ps0, executed_action)
                reward, post_obs, done = reward_obs(
                    ps0, ps1, executed_action)
                next_state = state.replace(
                    pipeline_state=ps1, obs=post_obs, reward=reward, done=done)

                reset_st = reset_state(sys, reset_key)
                out_state = jax.tree.map(
                    lambda r, n: jnp.where(done, r, n), reset_st, next_state)

                return out_state, (reward, done)

            final_state, (rewards, dones) = jax.lax.scan(
                scan_body, init_state, step_keys)
            return final_state, (rewards, dones)

        @jax.jit
        def _rollout_scan_det_horizon(
                sys, action_gain, action_noise_std, packet_loss_prob,
                burst_prob, burst_std, policy_params, context_params,
                belief_vec, init_state, step_keys, warmup, episode_horizon):
            """Eval rollout with forced reset at fixed episode boundaries."""

            def scan_body(carry, step_key):
                state, step_in_episode = carry
                noise_key, reset_key = jax.random.split(step_key)
                pre_obs = state.obs
                policy = nnx.merge(policy_graphdef, policy_params)

                if has_context:
                    ctx_net = nnx.merge(context_graphdef, context_params)
                    ep = ctx_net(pre_obs[None])
                    ep = jnp.where(warmup, jnp.zeros_like(ep), ep)
                    if belief_vec is not None:
                        b = jnp.where(warmup, jnp.zeros_like(belief_vec),
                                      belief_vec)
                        ep = jnp.concatenate([ep, b[None, :]], axis=-1)
                    action = policy.deterministic(pre_obs[None], ep)
                elif has_direct_context:
                    ep = jnp.where(
                        warmup, jnp.zeros_like(belief_vec), belief_vec)
                    action = policy.deterministic(
                        pre_obs[None], ep[None, :])
                else:
                    action = policy.deterministic(pre_obs[None])
                action = action[0]
                executed_action = apply_action_disturbance(
                    action, noise_key, action_gain, action_noise_std,
                    packet_loss_prob, burst_prob, burst_std)

                ps0 = state.pipeline_state
                ps1 = physics_step(sys, ps0, executed_action)
                reward, post_obs, done = reward_obs(
                    ps0, ps1, executed_action)
                next_state = state.replace(
                    pipeline_state=ps1, obs=post_obs, reward=reward, done=done)

                horizon_done = step_in_episode + 1 >= episode_horizon
                episode_done = jnp.logical_or(done > 0.5, horizon_done)
                reset_st = reset_state(sys, reset_key)
                out_state = jax.tree.map(
                    lambda r, n: jnp.where(episode_done, r, n),
                    reset_st, next_state)
                next_step = jnp.where(episode_done, 0, step_in_episode + 1)

                return (out_state, next_step), (reward, done, horizon_done)

            init_carry = (init_state, jnp.asarray(0, dtype=jnp.int32))
            (final_state, _), outputs = jax.lax.scan(
                scan_body, init_carry, step_keys)
            return final_state, outputs

        @jax.jit
        def _rollout_scan_det_recurrent_horizon(
                sys, action_gain, action_noise_std, packet_loss_prob,
                burst_prob, burst_std, policy_params, context_params,
                init_hidden, init_previous_action, init_state, step_keys,
                episode_horizon):
            """Strict-horizon ESCP eval with causal recurrent resets."""
            policy = nnx.merge(policy_graphdef, policy_params)
            context_model = nnx.merge(
                recurrent_context_graphdef, context_params,
                recurrent_context_non_params)

            def scan_body(carry, step_key):
                state, hidden, previous_action, step_in_episode = carry
                process_noise_key, reset_key = jax.random.split(step_key)
                pre_obs = state.obs
                next_hidden, context = context_model.step(
                    hidden, pre_obs[None], previous_action[None])
                action = policy.deterministic(
                    pre_obs[None], context)[0]
                executed_action = apply_action_disturbance(
                    action, process_noise_key, action_gain,
                    action_noise_std, packet_loss_prob, burst_prob,
                    burst_std)

                ps0 = state.pipeline_state
                ps1 = physics_step(sys, ps0, executed_action)
                reward, post_obs, done = reward_obs(
                    ps0, ps1, executed_action)
                next_state = state.replace(
                    pipeline_state=ps1, obs=post_obs,
                    reward=reward, done=done)
                horizon_done = step_in_episode + 1 >= episode_horizon
                episode_done = jnp.logical_or(done > 0.5, horizon_done)
                reset_st = reset_state(sys, reset_key)
                out_state = jax.tree.map(
                    lambda reset, current: jnp.where(
                        episode_done, reset, current),
                    reset_st, next_state)
                out_hidden = jnp.where(
                    episode_done, jnp.zeros_like(next_hidden), next_hidden)
                out_previous_action = jnp.where(
                    episode_done, jnp.zeros_like(action), action)
                next_step = jnp.where(
                    episode_done, 0, step_in_episode + 1)
                return (
                    out_state, out_hidden, out_previous_action, next_step
                ), (reward, done, horizon_done)

            initial = (
                init_state, init_hidden, init_previous_action,
                jnp.asarray(0, dtype=jnp.int32))
            final, outputs = jax.lax.scan(
                scan_body, initial, step_keys)
            return final, outputs

        @jax.jit
        def _rollout_scan_det_adaptive_horizon(
                sys, action_gain, action_noise_std, packet_loss_prob,
                burst_prob, burst_std, policy_params, critic_params,
                context_params,
                adaptation_state, oracle_latent, init_state, reset_keys,
                episode_horizon, context_source, advantage_enabled,
                advantage_margin, advantage_lcb_scale):
            """Strict-horizon mean-policy eval with causal latent updates."""
            context_model = nnx.merge(
                transition_context_graphdef, context_params)
            initial_adapt = context_model.initial_state()

            def scan_body(carry, step_key):
                state, adapt, step_in_episode = carry
                noise_key, reset_key = jax.random.split(step_key)
                policy = nnx.merge(policy_graphdef, policy_params)
                pre_obs = state.obs
                context = adaptive_context(
                    context_model, adapt, oracle_latent, context_source)
                if has_advantage_critic:
                    critic = nnx.merge(critic_graphdef, critic_params)
                    action, advantage, advantage_gate = (
                        advantage_gated_action(
                            policy, critic, pre_obs[None], context[None],
                            enabled=advantage_enabled,
                            margin=advantage_margin,
                            lcb_scale=advantage_lcb_scale))
                    action = action[0]
                    advantage = advantage[0]
                    advantage_gate = advantage_gate[0]
                else:
                    action = policy.deterministic(
                        pre_obs[None], context[None])[0]
                    advantage = jnp.asarray(0.0, dtype=action.dtype)
                    advantage_gate = jnp.asarray(1.0, dtype=action.dtype)
                executed_action = apply_action_disturbance(
                    action, noise_key, action_gain, action_noise_std,
                    packet_loss_prob, burst_prob, burst_std)

                ps0 = state.pipeline_state
                ps1 = physics_step(sys, ps0, executed_action)
                reward, post_obs, done = reward_obs(
                    ps0, ps1, executed_action)
                next_state = state.replace(
                    pipeline_state=ps1, obs=post_obs,
                    reward=reward, done=done)
                next_adapt, error, _, _ = context_model.observe(
                    adapt, pre_obs, action, reward, post_obs, done,
                    enable_reset=True)

                horizon_done = step_in_episode + 1 >= episode_horizon
                episode_done = jnp.logical_or(done > 0.5, horizon_done)
                reset_st = reset_state(sys, reset_key)
                out_state = jax.tree.map(
                    lambda r, n: jnp.where(episode_done, r, n),
                    reset_st, next_state)
                out_adapt = jax.tree.map(
                    lambda initial, current: jnp.where(
                        episode_done, initial, current),
                    initial_adapt, next_adapt)
                next_step = jnp.where(
                    episode_done, 0, step_in_episode + 1)
                return (
                    out_state, out_adapt, next_step), (
                    reward, done, horizon_done, context[-1], error,
                    advantage, advantage_gate)

            init = (
                init_state, adaptation_state,
                jnp.asarray(0, dtype=jnp.int32))
            (final_state, final_adapt, _), outputs = jax.lax.scan(
                scan_body, init, reset_keys)
            return final_state, final_adapt, outputs

        self._rollout_scan = _rollout_scan
        self._rollout_scan_det = _rollout_scan_det
        self._rollout_scan_det_horizon = _rollout_scan_det_horizon
        if has_recurrent_context:
            self._rollout_scan_recurrent = _rollout_scan_recurrent
            self._rollout_scan_det_recurrent_horizon = (
                _rollout_scan_det_recurrent_horizon)
        if has_transition_context:
            self._rollout_scan_adaptive = _rollout_scan_adaptive
            self._rollout_scan_det_adaptive_horizon = (
                _rollout_scan_det_adaptive_horizon)
        self._has_context = has_context
        self._has_transition_context = has_transition_context
        self._has_recurrent_context = has_recurrent_context
        self._recurrent_context_graphdef = recurrent_context_graphdef
        self._recurrent_context_non_params = recurrent_context_non_params
        self._has_direct_policy_context = has_direct_context

    def rollout(self, policy_params, n_steps: int, rng_key,
                context_params=None, belief_vec=None, warmup=False,
                continue_state=False, return_executed_action=False):
        """Run n_steps using the pre-built scan rollout.

        Returns JAX arrays (stay on GPU) + episode rewards (CPU).
        The caller (collect_samples) can push JAX arrays directly to the
        GPU-native replay buffer via push_batch_jax — zero CPU transfer.

        Args:
            policy_params: nnx.State(agent.policy, nnx.Param)
            n_steps: number of steps
            rng_key: PRNG key
            context_params: optional nnx.State(agent.context_net, nnx.Param)

        Returns:
            (obs, act, rew, nobs, done): JAX arrays [n_steps, ...].
                If ``return_executed_action`` is true, the transition tuple
                also includes the disturbed action applied to physics.
            ep_rewards: list of float (computed on CPU from done mask)
        """
        if self._has_direct_policy_context and belief_vec is None:
            raise ValueError(
                "direct conditioned rollout requires belief_vec")
        rng_key, init_key, roll_key = jax.random.split(rng_key, 3)
        init_state = (
            self._state
            if bool(continue_state) and self._state is not None
            else self._reset_fn(self._current_sys, init_key))
        keys = jax.random.split(roll_key, n_steps)
        (action_gain, action_noise_std, packet_loss_prob,
         burst_prob, burst_std) = self.action_disturbance_params()

        # Single JIT call for all N steps
        final_state, (
            obs, act, rew, nobs, done, executed_action
        ) = self._rollout_scan(
            self._current_sys, action_gain, action_noise_std,
            packet_loss_prob, burst_prob, burst_std, policy_params,
            context_params, belief_vec, init_state, keys, jnp.asarray(warmup))

        # Episode rewards need CPU for Python-level segmentation
        rew_np = np.array(rew)
        done_np = np.array(done)
        ep_rewards = []
        ep_r = 0.0
        for i in range(n_steps):
            ep_r += rew_np[i]
            if done_np[i] > 0.5:
                ep_rewards.append(float(ep_r))
                ep_r = 0.0

        # Update step counter and trigger task switch for NEXT rollout
        self._step_counter += n_steps
        self._check_switch()   # ← switches _current_sys so next rollout uses new task
        self._state = final_state
        # Return JAX arrays (obs, act, rew, nobs, done stay on GPU)
        transitions = (obs, act, rew, nobs, done)
        if return_executed_action:
            transitions = transitions + (executed_action,)
        return transitions, ep_rewards

    def rollout_recurrent(
            self, policy_params, context_params, recurrent_hidden,
            previous_action, n_steps: int, rng_key, warmup=False,
            context_noise_sigma=0.0, continue_state=False):
        """Run a stochastic ESCP rollout and preserve its recurrent carry."""
        if not getattr(self, "_has_recurrent_context", False):
            raise RuntimeError("recurrent rollout was not built")
        rng_key, init_key, rollout_key = jax.random.split(rng_key, 3)
        init_state = (
            self._state
            if bool(continue_state) and self._state is not None
            else self._reset_fn(self._current_sys, init_key))
        keys = jax.random.split(rollout_key, n_steps)
        (action_gain, action_noise_std, packet_loss_prob,
         burst_prob, burst_std) = self.action_disturbance_params()
        (final_state, final_hidden, final_previous_action), transitions = (
            self._rollout_scan_recurrent(
                self._current_sys, action_gain, action_noise_std,
                packet_loss_prob, burst_prob, burst_std, policy_params,
                context_params, recurrent_hidden, previous_action,
                init_state, keys, jnp.asarray(warmup),
                jnp.asarray(context_noise_sigma, dtype=jnp.float32)))
        obs, act, rew, nobs, done, _ = transitions

        rew_np = np.asarray(rew)
        done_np = np.asarray(done)
        episode_rewards = []
        episode_reward = 0.0
        for idx in range(n_steps):
            episode_reward += float(rew_np[idx])
            if done_np[idx] > 0.5:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0

        self._step_counter += n_steps
        self._check_switch()
        self._state = final_state
        return (
            obs, act, rew, nobs, done
        ), episode_rewards, final_hidden, final_previous_action

    def rollout_adaptive(
            self, policy_params, context_params, adaptation_state,
            oracle_latent, n_steps: int, rng_key, warmup=False,
            critic_params=None, context_source=2, advantage_enabled=False,
            advantage_margin=0.0, advantage_lcb_scale=1.0,
            continue_state=False):
        """Run a BAPR-v2 rollout and return per-transition causal contexts."""
        rng_key, init_key, roll_key = jax.random.split(rng_key, 3)
        init_state = (
            self._state
            if bool(continue_state) and self._state is not None
            else self._reset_fn(self._current_sys, init_key))
        keys = jax.random.split(roll_key, n_steps)
        (action_gain, action_noise_std, packet_loss_prob,
         burst_prob, burst_std) = self.action_disturbance_params()
        final_state, final_adapt, transitions = self._rollout_scan_adaptive(
            self._current_sys, action_gain, action_noise_std,
            packet_loss_prob, burst_prob, burst_std, policy_params,
            critic_params, context_params, adaptation_state, oracle_latent,
            init_state, keys,
            jnp.asarray(warmup), jnp.asarray(context_source, jnp.int32),
            jnp.asarray(advantage_enabled, jnp.bool_),
            jnp.asarray(advantage_margin, jnp.float32),
            jnp.asarray(advantage_lcb_scale, jnp.float32))
        (obs, act, rew, nobs, done, context, next_context, error,
         advantage, advantage_gate) = transitions

        rew_np = np.asarray(rew)
        done_np = np.asarray(done)
        ep_rewards = []
        ep_reward = 0.0
        for idx in range(n_steps):
            ep_reward += float(rew_np[idx])
            if done_np[idx] > 0.5:
                ep_rewards.append(ep_reward)
                ep_reward = 0.0

        self._step_counter += n_steps
        self._check_switch()
        self._state = final_state
        return (
            obs, act, rew, nobs, done, context, next_context, error,
            advantage, advantage_gate
        ), ep_rewards, final_adapt

    def eval_rollout_recurrent(
            self, policy_params, context_params, n_steps: int, rng_key,
            episode_horizon: int):
        """Fast deterministic ESCP eval with strict recurrent boundaries."""
        if not getattr(self, "_has_recurrent_context", False):
            raise RuntimeError("recurrent eval rollout was not built")
        from flax import nnx

        rng_key, init_key, reset_key = jax.random.split(rng_key, 3)
        sys = self._current_sys
        init_state = self._reset_fn(sys, init_key)
        step_keys = jax.random.split(reset_key, n_steps)
        context_model = nnx.merge(
            self._recurrent_context_graphdef, context_params,
            self._recurrent_context_non_params)
        init_hidden = context_model.initial_hidden((1,))
        init_previous_action = jnp.zeros(
            (self.act_dim,), dtype=jnp.float32)
        (action_gain, action_noise_std, packet_loss_prob,
         burst_prob, burst_std) = self.action_disturbance_params()
        horizon = jnp.asarray(int(episode_horizon), dtype=jnp.int32)
        _, (rewards, dones, _) = self._rollout_scan_det_recurrent_horizon(
            sys, action_gain, action_noise_std, packet_loss_prob,
            burst_prob, burst_std, policy_params, context_params,
            init_hidden, init_previous_action, init_state, step_keys,
            horizon)
        return np.asarray(rewards), np.asarray(dones)

    def eval_rollout(self, policy_params, n_steps: int, rng_key,
                     context_params=None, belief_vec=None, warmup=False,
                     episode_horizon=None):
        """Deterministic eval rollout — does NOT update step counter or switch task.

        Uses tanh(mean) policy (no exploration noise). ~10-50x faster than the
        sequential evaluate() loop because all steps run in one GPU scan call.

        Args:
            policy_params: nnx.State(agent.policy, nnx.Param)
            n_steps: total steps to run (e.g. n_episodes * max_episode_steps)
            rng_key: PRNG key (only used for auto-reset state sampling)
            context_params: optional nnx.State(agent.context_net, nnx.Param)

        Returns:
            (rew_np, done_np): per-step reward and done arrays [n_steps]
        """
        if self._has_direct_policy_context and belief_vec is None:
            raise ValueError(
                "direct conditioned eval requires belief_vec")
        rng_key, init_key, reset_key = jax.random.split(rng_key, 3)
        sys = self._current_sys
        init_state = self._reset_fn(sys, init_key)
        step_keys = jax.random.split(reset_key, n_steps)
        (action_gain, action_noise_std, packet_loss_prob,
         burst_prob, burst_std) = self.action_disturbance_params()

        if episode_horizon is None:
            _, (rew_jax, done_jax) = self._rollout_scan_det(
                sys, action_gain, action_noise_std, packet_loss_prob,
                burst_prob, burst_std, policy_params, context_params,
                belief_vec, init_state, step_keys, jnp.asarray(warmup))
        else:
            horizon = jnp.asarray(int(episode_horizon), dtype=jnp.int32)
            _, (rew_jax, done_jax, _) = self._rollout_scan_det_horizon(
                sys, action_gain, action_noise_std, packet_loss_prob,
                burst_prob, burst_std, policy_params, context_params,
                belief_vec, init_state, step_keys, jnp.asarray(warmup),
                horizon)

        return np.array(rew_jax), np.array(done_jax)

    def eval_rollout_adaptive(
            self, policy_params, context_params, adaptation_state,
            oracle_latent, n_steps: int, rng_key, episode_horizon: int,
            critic_params=None, context_source=2, advantage_enabled=False,
            advantage_margin=0.0, advantage_lcb_scale=1.0):
        """Fast stationary BAPR-v2 eval using the same causal update as train."""
        rng_key, init_key, reset_key = jax.random.split(rng_key, 3)
        sys = self._current_sys
        init_state = self._reset_fn(sys, init_key)
        step_keys = jax.random.split(reset_key, n_steps)
        (action_gain, action_noise_std, packet_loss_prob,
         burst_prob, burst_std) = self.action_disturbance_params()
        horizon = jnp.asarray(int(episode_horizon), dtype=jnp.int32)
        _, _, outputs = self._rollout_scan_det_adaptive_horizon(
            sys, action_gain, action_noise_std, packet_loss_prob,
            burst_prob, burst_std, policy_params, critic_params,
            context_params, adaptation_state, oracle_latent, init_state,
            step_keys, horizon,
            jnp.asarray(context_source, jnp.int32),
            jnp.asarray(advantage_enabled, jnp.bool_),
            jnp.asarray(advantage_margin, jnp.float32),
            jnp.asarray(advantage_lcb_scale, jnp.float32))
        rewards, dones, _, gates, errors, advantages, advantage_gates = outputs
        diagnostics = {
            "gate_mean": float(jnp.mean(gates)),
            "error_mean": float(jnp.mean(errors)),
            "advantage_mean": float(jnp.mean(advantages)),
            "advantage_gate_mean": float(jnp.mean(advantage_gates)),
        }
        return np.asarray(rewards), np.asarray(dones), diagnostics
