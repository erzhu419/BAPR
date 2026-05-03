"""Discrete-mode piecewise-stationary env for BAPR's design sweet spot.

Provides BraxNonstationaryEnv-compatible API with 4 properties (see
BAPR_ENV_DESIGN.md for rationale):

  A. K=4 discrete modes per env, each requires a qualitatively different
     optimal policy (multi-param combinations: gravity × body_mass × dof_damping).
  B. Long metastable periods (default 60 iters dwell, far above current 10).
  C. Random switching: dwell time ~ Exponential(mean), next mode ~ Uniform(others).
  D. Catastrophic mismatch cost emerges naturally from physics: wrong-mode
     policy → wrong dynamics → low/negative reward (bonus reward shaping
     available via reward_shaping=True).

Used via configs/default.py: env_type='discrete_mode'.
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List

from jax_experiments.envs.brax_env import (
    BraxNonstationaryEnv, RAND_PARAMS_MAP, BRAX_ENV_MAP)


# Per-env semantic mode definitions.
# Each value is a dict {param_name: multiplier}. Empty dict = default (normal).
# Multipliers are MULTIPLICATIVE (param = base * mult).
MODE_DEFINITIONS: Dict[str, Dict[str, Dict[str, float]]] = {
    'HalfCheetah-v2': {
        'normal':       {},                                       # baseline
        'heavy_low_g':  {'body_mass': 2.0, 'gravity': 0.6},       # heavy + weak g
        'light_high_g': {'body_mass': 0.5, 'gravity': 1.5},       # light + strong g
        'stiff_joints': {'dof_damping': 3.0},                     # rigid
    },
    'Hopper-v2': {
        'normal':      {},
        'slippery':    {'dof_damping': 0.3},                      # less damping
        'heavy_load':  {'body_mass': 1.5, 'gravity': 1.3},
        'low_grav':    {'gravity': 0.5},
    },
    'Walker2d-v2': {
        'normal':      {},
        'slippery':    {'dof_damping': 0.3},
        'heavy_load':  {'body_mass': 1.5, 'gravity': 1.3},
        'low_grav':    {'gravity': 0.5},
    },
    'Ant-v2': {
        'normal':       {},
        'heavy_low_g':  {'body_mass': 2.0, 'gravity': 0.6},
        'light_high_g': {'body_mass': 0.5, 'gravity': 1.5},
        'stiff_joints': {'dof_damping': 3.0},
    },
}


# Milder variant presets for environments where the default modes are too
# extreme (e.g. Hopper/Walker2d under low_grav g*0.5 / slippery d*0.3 cause
# uniform plateau ~300-500 across all algos — see paper sec:env_difficulty).
# Each variant is keyed by env_name and provides 4 alternative modes.
# Selected via train.py --mode_variant <name>; default 'orig' = MODE_DEFINITIONS.
MODE_VARIANTS: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {
    'Hopper-v2': {
        'g_mild': {
            'normal':    {},
            'g_low':     {'gravity': 0.85},
            'g_high':    {'gravity': 1.15},
            'g_higher':  {'gravity': 1.20},
        },
        'd_mild': {
            'normal':    {},
            'd_low':     {'dof_damping': 0.7},
            'd_high':    {'dof_damping': 1.3},
            'd_higher':  {'dof_damping': 1.5},
        },
        'm_mild': {
            'normal':    {},
            'm_low':     {'body_mass': 0.85},
            'm_high':    {'body_mass': 1.15},
            'm_higher':  {'body_mass': 1.30},
        },
        'mixed_mild': {
            'normal':       {},
            'g_low_d_high': {'gravity': 0.9, 'dof_damping': 1.1},
            'g_high_d_low': {'gravity': 1.1, 'dof_damping': 0.9},
            'm_high':       {'body_mass': 1.20},
        },
    },
    'Walker2d-v2': {
        'g_mild': {
            'normal':    {},
            'g_low':     {'gravity': 0.85},
            'g_high':    {'gravity': 1.15},
            'g_higher':  {'gravity': 1.20},
        },
        'd_mild': {
            'normal':    {},
            'd_low':     {'dof_damping': 0.7},
            'd_high':    {'dof_damping': 1.3},
            'd_higher':  {'dof_damping': 1.5},
        },
        'm_mild': {
            'normal':    {},
            'm_low':     {'body_mass': 0.85},
            'm_high':    {'body_mass': 1.15},
            'm_higher':  {'body_mass': 1.30},
        },
        'mixed_mild': {
            'normal':       {},
            'g_low_d_high': {'gravity': 0.9, 'dof_damping': 1.1},
            'g_high_d_low': {'gravity': 1.1, 'dof_damping': 0.9},
            'm_high':       {'body_mass': 1.20},
        },
    },
}


def _resolve_mode_definitions(env_name: str, variant: str = 'orig'):
    """Return the {mode_name: param_dict} dict for the (env, variant) pair.

    variant='orig' → MODE_DEFINITIONS[env_name] (back-compat)
    otherwise     → MODE_VARIANTS[env_name][variant]
    """
    if variant == 'orig' or variant is None:
        return MODE_DEFINITIONS.get(env_name)
    if env_name not in MODE_VARIANTS:
        raise ValueError(f"No MODE_VARIANTS for env={env_name!r}")
    if variant not in MODE_VARIANTS[env_name]:
        raise ValueError(f"variant {variant!r} not in MODE_VARIANTS[{env_name!r}]; "
                         f"available: {list(MODE_VARIANTS[env_name].keys())}")
    return MODE_VARIANTS[env_name][variant]


class DiscreteModePiecewiseEnv(BraxNonstationaryEnv):
    """Piecewise-stationary brax env with K discrete semantic modes.

    Inherits build_rollout_fn / rollout / eval_rollout / step / reset from
    BraxNonstationaryEnv. Overrides task management:
      - sample_tasks returns the K mode dicts (n_tasks arg ignored)
      - set_nonstationary_para is a no-op (dwell schedule is internal)
      - _check_switch uses random Exponential dwell + uniform next-mode
    """

    def __init__(self, env_name: str, mode_keys: List[str] = None,
                 mean_dwell_iters: int = 60,
                 steps_per_iter: int = 4000,
                 dwell_distribution: str = 'exponential',
                 reward_shaping: bool = False,
                 expected_rewards: Dict[str, float] = None,
                 reward_floor_frac: float = 0.3,
                 reward_floor_penalty: float = -50.0,
                 seed: int = 0, backend: str = 'spring',
                 mode_variant: str = 'orig'):
        """
        Args:
            env_name: brax env name (e.g. 'HalfCheetah-v2')
            mode_keys: list of mode names to use; defaults to all from MODE_DEFINITIONS
            mean_dwell_iters: average dwell time per mode in iters (40-100 recommended)
            steps_per_iter: brax steps per training iter (must match samples_per_iter)
            dwell_distribution: 'exponential' (random) or 'fixed' (deterministic)
            reward_shaping: if True, apply reward floor penalty for mode-mismatch
            expected_rewards: per-mode expected reward (for floor detection)
            reward_floor_frac: trigger penalty when reward < expected * frac
            reward_floor_penalty: extra penalty when triggered
            seed, backend: forwarded to brax
        """
        # All possible params we may modulate across modes (union)
        # Use a placeholder log_scale of 0 (modes provide explicit multipliers)
        all_params = ['gravity', 'body_mass', 'dof_damping']
        super().__init__(env_name, rand_params=all_params,
                         log_scale_limit=0.0, seed=seed, backend=backend)

        # Resolve modes for this env (variant='orig' falls back to MODE_DEFINITIONS)
        env_modes = _resolve_mode_definitions(env_name, mode_variant)
        if env_modes is None:
            raise ValueError(
                f"No mode definitions for env_name={env_name}, "
                f"variant={mode_variant!r}.")
        self.mode_variant = mode_variant
        if mode_keys is None:
            mode_keys = list(env_modes.keys())
        for k in mode_keys:
            if k not in env_modes:
                raise ValueError(
                    f"mode_key {k!r} not in MODE_DEFINITIONS[{env_name!r}]")
        self.mode_keys = mode_keys
        self.modes = [env_modes[k] for k in mode_keys]
        self.K = len(self.modes)

        # Switching schedule (Property B + C)
        self.mean_dwell_iters = mean_dwell_iters
        self.dwell_steps_mean = mean_dwell_iters * steps_per_iter
        self.dwell_distribution = dwell_distribution

        # Optional reward shaping (Property D)
        self.reward_shaping = reward_shaping
        self.expected_rewards = expected_rewards or {}
        self.reward_floor_frac = reward_floor_frac
        self.reward_floor_penalty = reward_floor_penalty

        # Pre-compute brax sys for each mode (multi-param replacements)
        self._mode_sys = []
        for mode_dict in self.modes:
            replacements = {}
            for param, mult in mode_dict.items():
                key = RAND_PARAMS_MAP.get(param, param)
                base = self._base_values.get(param)
                if base is None:
                    continue
                replacements[key] = jnp.asarray(
                    np.array(base) * float(mult), dtype=jnp.float32)
            self._mode_sys.append(self.base_sys.tree_replace(replacements))

        # Init state
        self._mode_rng = np.random.RandomState(seed + 99)
        self.current_task_id = 0
        self._set_sys(self._mode_sys[0])
        self._step_counter = 0
        self._next_switch_step = self._sample_dwell()

        # Task switching trace (for logging)
        self._switch_history = [(0, 0)]  # list of (step, new_mode)

    # ---- Property B + C: random dwell + random next-mode ----

    def _sample_dwell(self) -> int:
        """Sample dwell time for current mode in env steps."""
        if self.dwell_distribution == 'exponential':
            return int(self._mode_rng.exponential(self.dwell_steps_mean))
        elif self.dwell_distribution == 'fixed':
            return self.dwell_steps_mean
        else:
            raise ValueError(f"unknown dwell_distribution: {self.dwell_distribution!r}")

    def _sample_next_mode(self) -> int:
        """Sample uniformly over OTHER modes (never stay in current)."""
        if self.K == 1:
            return 0
        others = [i for i in range(self.K) if i != self.current_task_id]
        return int(self._mode_rng.choice(others))

    # ---- Override BraxNonstationaryEnv.{_check_switch, sample_tasks, set_nonstationary_para} ----

    def _check_switch(self):
        """Check whether to switch mode. Called from step()."""
        if self._step_counter >= self._next_switch_step:
            old_mode = self.current_task_id
            self.current_task_id = self._sample_next_mode()
            self._set_sys(self._mode_sys[self.current_task_id])
            self._next_switch_step = self._step_counter + self._sample_dwell()
            self._switch_history.append((self._step_counter, self.current_task_id))

    def sample_tasks(self, n_tasks: int) -> List[Dict]:
        """Returns K mode descriptors (n_tasks arg ignored).

        Each dict carries only numeric fields so train.py's set_task path
        does not choke on string values via jnp.array. Use mode_id (int) as
        the canonical handle; mode_name is recoverable via self.mode_keys.
        """
        return [{'mode_id': i, **self.modes[i]} for i in range(self.K)]

    def set_task(self, task: Dict):
        """Switch to a pre-computed mode by mode_id.

        Overrides BraxNonstationaryEnv.set_task to bypass jnp.array on
        param multipliers (modes are pre-baked in _mode_sys).
        """
        if 'mode_id' in task:
            mode_id = int(task['mode_id'])
        else:
            mode_id = 0  # default to first mode if unspecified
        self.current_task_id = mode_id
        self._set_sys(self._mode_sys[mode_id])

    def set_nonstationary_para(self, tasks, changing_period=None,
                                changing_interval=None):
        """No-op: dwell schedule is internal, not driven by external period.

        We accept (tasks, changing_period, changing_interval) for API compat
        with train.py but ignore them. The dwell schedule was set in __init__
        via mean_dwell_iters.
        """
        # Optional: respect a runtime override of mean_dwell if changing_period
        # is supplied (interpreted as mean dwell time). For now just keep init.
        pass

    # ---- Property D (optional): reward shaping for catastrophic mismatch ----

    def step(self, action):
        """Override step to apply optional reward shaping."""
        self._step_counter += 1
        self._check_switch()
        action_jax = jnp.array(action)
        self._state = self._step_fn(self._current_sys, self._state, action_jax)
        reward = float(self._state.reward)

        # Reward shaping (Property D, optional): penalize when reward
        # collapses below the per-mode expected level — a behavioral cue
        # that policy is mismatched to current mode.
        if self.reward_shaping:
            mode_name = self.mode_keys[self.current_task_id]
            expected = self.expected_rewards.get(mode_name, None)
            if expected is not None and expected > 0:
                if reward < expected * self.reward_floor_frac:
                    reward += self.reward_floor_penalty

        return (np.array(self._state.obs), reward,
                bool(self._state.done), {'mode': self.current_task_id})

    # ---- Public introspection ----

    @property
    def num_modes(self) -> int:
        return self.K

    @property
    def current_mode_name(self) -> str:
        return self.mode_keys[self.current_task_id]

    def get_switch_history(self):
        """Returns list of (step, new_mode_id) switches that happened."""
        return list(self._switch_history)
