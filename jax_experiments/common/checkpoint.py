"""Checkpoint save/load for resumable training.

Saves:
  - Model params (policy, critic, target_critic, context_net, etc.) via flax state dicts
  - Optimizer states
  - Replay buffer data
  - Training state (iteration, total_steps, log_alpha, update_count, logger data)
  - BOCD belief tracker state (for BAPR variants)
"""
import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx


def _to_numpy_tree(pytree):
    """Convert a JAX pytree (nnx.State / optax state) to numpy for serialization."""
    return jax.tree.map(lambda x: np.array(x) if hasattr(x, 'shape') else x, pytree)


def _to_jax_tree(pytree):
    """Convert numpy pytree back to JAX arrays."""
    return jax.tree.map(lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, pytree)


def _patch_flax_variablestate_unpickle():
    """Allow Flax NNX to read checkpoints saved by nearby NNX builds."""
    try:
        from flax.nnx import variablelib
        raw_default_state = nnx.Param(0).__getstate__()

        def _patch_raw_value_variable(cls):
            """Make Flax Variable/Param unpickle states from nearby NNX builds."""
            if cls is None or getattr(cls, "_bapr_raw_value_pickle_patch", False):
                return
            orig_setstate = getattr(cls, "__setstate__", None)
            supported_slots = set(getattr(cls, "__slots__", ()) or ())
            if "raw_value" not in supported_slots:
                return

            def _compat_setstate(self, state):
                if (isinstance(state, tuple) and len(state) == 2 and
                        isinstance(state[1], dict)):
                    state = state[1]
                if isinstance(state, dict):
                    state = dict(state)
                    if "raw_value" not in state and "_raw_value" in state:
                        state["raw_value"] = state["_raw_value"]
                    if "raw_value" not in state and "value" in state:
                        state["raw_value"] = state["value"]
                    if "_trace_state" not in state:
                        state["_trace_state"] = raw_default_state.get(
                            "_trace_state")
                    if "_var_metadata" not in state:
                        state["_var_metadata"] = raw_default_state.get(
                            "_var_metadata", {})
                if orig_setstate is not None:
                    return orig_setstate(self, state)
                if not isinstance(state, dict):
                    raise TypeError(
                        f"Unsupported Variable pickle state: {type(state)}")
                object.__setattr__(self, "raw_value", state["raw_value"])
                object.__setattr__(self, "_trace_state",
                                   state.get("_trace_state"))
                object.__setattr__(self, "_var_metadata",
                                   state.get("_var_metadata", {}))

            cls.__setstate__ = _compat_setstate
            cls._bapr_raw_value_pickle_patch = True

        _patch_raw_value_variable(getattr(variablelib, "Variable", None))
        _patch_raw_value_variable(getattr(variablelib, "Param", None))

        cls = variablelib.VariableState
        orig = getattr(cls, "__setstate__", None)
        if getattr(cls, "_bapr_legacy_pickle_patch", False):
            return

        # Flax 0.10 VariableState has slots (type, value, _var_metadata) and
        # no __setstate__. Pickle probes __setstate__ before slots are
        # restored; its __getattr__ then raises while _var_metadata is absent.
        if orig is None:
            orig_getattr = getattr(cls, "__getattr__", None)

            def _compat_getattr(self, name):
                try:
                    if orig_getattr is not None:
                        return orig_getattr(self, name)
                except AttributeError as exc:
                    if "_var_metadata" in str(exc):
                        raise AttributeError(name) from None
                    raise
                raise AttributeError(name)

            supported_slots = set(getattr(cls, "__slots__", ()) or ())
            dynamic_attrs = not supported_slots
            default_state = nnx.Param(0).__getstate__()

            def _safe_setattr(obj, name, value):
                try:
                    object.__setattr__(obj, name, value)
                except AttributeError:
                    pass

            def _supports_attr(name):
                return dynamic_attrs or name in supported_slots

            def _compat_setstate(self, state):
                if (isinstance(state, tuple) and len(state) == 2 and
                        isinstance(state[1], dict)):
                    state = state[1]
                if not isinstance(state, dict):
                    raise TypeError(
                        f"Unsupported VariableState pickle state: {type(state)}")
                value = state.get("_raw_value",
                                  state.get("raw_value", state.get("value")))
                metadata = {
                    k: v for k, v in state.items()
                    if k not in ("type", "value", "raw_value", "_raw_value",
                                 "_var_metadata", "_trace_state")
                }
                legacy_metadata = state.get("_var_metadata", {}) or {}
                if isinstance(legacy_metadata, dict):
                    metadata.update({
                        k: v for k, v in legacy_metadata.items()
                        if k != "type"
                    })
                for hook_name in (
                    "get_value_hooks",
                    "set_value_hooks",
                    "create_value_hooks",
                    "add_axis_hooks",
                    "remove_axis_hooks",
                ):
                    metadata.setdefault(hook_name, ())
                var_type = state.get("type", nnx.Param)
                if _supports_attr("type"):
                    _safe_setattr(self, "type", var_type)
                if _supports_attr("value"):
                    _safe_setattr(self, "value", value)
                elif "_raw_value" in supported_slots:
                    _safe_setattr(self, "_raw_value", value)
                if "_trace_state" in supported_slots:
                    _safe_setattr(
                        self, "_trace_state",
                        state.get("_trace_state", default_state.get("_trace_state")))
                if "_var_metadata" in supported_slots:
                    _safe_setattr(self, "_var_metadata", metadata)
                for k, v in metadata.items():
                    if k in ("type", "value", "_raw_value", "_var_metadata"):
                        continue
                    if _supports_attr(k):
                        _safe_setattr(self, k, v)

            cls.__getattr__ = _compat_getattr
            cls.__setstate__ = _compat_setstate
            cls._bapr_legacy_pickle_patch = True
            return

        default_state = nnx.Param(0).__getstate__()

        def _compat_setstate(self, state):
            if (isinstance(state, tuple) and len(state) == 2 and
                    isinstance(state[1], dict)):
                state = state[1]
            if isinstance(state, dict):
                state = dict(state)
                if "raw_value" not in state and "_raw_value" in state:
                    state["raw_value"] = state["_raw_value"]
                if "raw_value" not in state and "value" in state:
                    state["raw_value"] = state["value"]
                if "_raw_value" not in state and "raw_value" in state:
                    state["_raw_value"] = state["raw_value"]
                if "_raw_value" not in state and "value" in state:
                    state["_raw_value"] = state["value"]
                if "_trace_state" not in state:
                    state["_trace_state"] = default_state.get("_trace_state")
                if "_var_metadata" not in state:
                    state["_var_metadata"] = default_state.get("_var_metadata", {})
                if isinstance(state.get("_var_metadata"), dict):
                    state["_var_metadata"] = dict(state["_var_metadata"])
                    state["_var_metadata"].setdefault(
                        "type", state.get("type", nnx.Param))
            return orig(self, state)

        cls.__setstate__ = _compat_setstate
        cls._bapr_legacy_pickle_patch = True
    except Exception:
        pass


def save_checkpoint(ckpt_dir: str, agent, replay_buffer, logger,
                    iteration: int, total_steps: int, algo: str):
    """Save a full training checkpoint to disk.

    Args:
        ckpt_dir: Directory to save checkpoint files.
        agent: Algorithm instance (SACBase, RESAC, ESCP, BAPR, or ablation).
        replay_buffer: ReplayBuffer instance.
        logger: Logger instance.
        iteration: Current iteration number.
        total_steps: Total environment steps collected.
        algo: Algorithm name string.
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Model params ---
    params = {}
    params['policy'] = _to_numpy_tree(nnx.state(agent.policy, nnx.Param))
    params['critic'] = _to_numpy_tree(nnx.state(agent.critic, nnx.Param))
    params['target_critic'] = _to_numpy_tree(nnx.state(agent.target_critic, nnx.Param))
    params['log_alpha'] = np.array(agent.log_alpha)
    params['update_count'] = agent.update_count

    # Optimizer states
    params['policy_opt_state'] = _to_numpy_tree(agent.policy_opt_state)
    params['critic_opt_state'] = _to_numpy_tree(agent.critic_opt_state)
    params['alpha_opt_state'] = _to_numpy_tree(agent.alpha_opt_state)

    # Context network (ESCP/BAPR variants)
    if hasattr(agent, 'context_net'):
        params['context_net'] = _to_numpy_tree(nnx.state(agent.context_net, nnx.Param))
        params['context_opt_state'] = _to_numpy_tree(agent.context_opt_state)

    # Belief tracker state (BAPR variants)
    if hasattr(agent, 'belief_tracker'):
        bt = agent.belief_tracker
        params['belief_state'] = {
            'belief': np.array(bt.belief),
        }
        if hasattr(agent, '_lw_baseline'):
            params['lw_baseline'] = agent._lw_baseline
        if hasattr(agent, '_current_weighted_lambda'):
            params['current_weighted_lambda'] = agent._current_weighted_lambda
        if hasattr(agent, '_reg_latched'):
            params['reg_latched'] = bool(agent._reg_latched)
        if hasattr(agent, '_train_reward_ema'):
            params['bapr_controller_state'] = {
                'train_reward_ema': agent._train_reward_ema,
                'train_reward_peak': agent._train_reward_peak,
                'eval_reward_ema': agent._eval_reward_ema,
                'eval_reward_peak': agent._eval_reward_peak,
                'last_train_reward_drop_frac': agent._last_train_reward_drop_frac,
                'last_eval_reward_drop_frac': agent._last_eval_reward_drop_frac,
                'last_perf_drop_frac': agent._last_perf_drop_frac,
                'controller_active': getattr(agent, '_controller_active', False),
                'controller_latched': getattr(agent, '_controller_latched', False),
                'controller_active_iters': getattr(
                    agent, '_controller_active_iters', 0),
                'controller_duration_released': getattr(
                    agent, '_controller_duration_released', False),
                'controller_cooldown_iters': getattr(
                    agent, '_controller_cooldown_iters', 0),
                'controller_exit_reason': getattr(
                    agent, '_controller_exit_reason', 0),
                'controller_signal': getattr(agent, '_controller_signal', 0.0),
                'controller_drawdown': getattr(agent, '_controller_drawdown', 0.0),
                'controller_low_disagreement': getattr(
                    agent, '_controller_low_disagreement', False),
                'last_train_reward_improve_frac': getattr(
                    agent, '_last_train_reward_improve_frac', 0.0),
                'last_eval_reward_improve_frac': getattr(
                    agent, '_last_eval_reward_improve_frac', 0.0),
                'controller_reg_multiplier': getattr(
                    agent, '_controller_reg_multiplier', 1.0),
                'controller_recent_multiplier': getattr(
                    agent, '_controller_recent_multiplier', 1.0),
                'controller_recent_add_frac': getattr(
                    agent, '_controller_recent_add_frac', 0.0),
                'controller_lcb_multiplier': getattr(
                    agent, '_controller_lcb_multiplier', 1.0),
                'controller_actor_update_multiplier': getattr(
                    agent, '_controller_actor_update_multiplier', 1.0),
            }

    # EMA policy (BAPR)
    if hasattr(agent, 'ema_policy'):
        params['ema_policy'] = _to_numpy_tree(nnx.state(agent.ema_policy, nnx.Param))

    # Performance gating state (BAPR)
    if hasattr(agent, '_eval_history'):
        params['eval_history'] = list(agent._eval_history)
        params['perf_gate_active'] = agent._perf_gate_active

    # Pseudo-label state (unsupervised variant)
    if hasattr(agent, '_current_pseudo_label'):
        params['pseudo_label'] = agent._current_pseudo_label
        params['last_lambda_was_high'] = agent._last_lambda_was_high

    # Save params
    with open(os.path.join(ckpt_dir, 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)

    # --- Replay buffer ---
    buf_data = replay_buffer.to_numpy()
    np.savez_compressed(os.path.join(ckpt_dir, 'replay_buffer.npz'), **buf_data)

    # --- Training state ---
    train_state = {
        'iteration': iteration,
        'total_steps': total_steps,
        'algo': algo,
        'logger_data': dict(logger.data),  # defaultdict -> dict for pickle
    }
    with open(os.path.join(ckpt_dir, 'train_state.pkl'), 'wb') as f:
        pickle.dump(train_state, f)

    print(f"  Checkpoint saved: iter={iteration}, steps={total_steps}, "
          f"buf={replay_buffer.size}")


def load_checkpoint(ckpt_dir: str, agent, replay_buffer, logger, algo: str,
                    load_replay_buffer: bool = True):
    """Load a training checkpoint. Returns (start_iteration, total_steps).

    Args:
        ckpt_dir: Directory containing checkpoint files.
        agent: Algorithm instance (already constructed with correct architecture).
        replay_buffer: Empty ReplayBuffer instance.
        logger: Logger instance.
        algo: Algorithm name string.
        load_replay_buffer: Whether to restore replay_buffer.npz. Evaluation
            callers can skip this because they only need model weights.

    Returns:
        (start_iteration, total_steps) to resume from.
    """
    params_path = os.path.join(ckpt_dir, 'params.pkl')
    buf_path = os.path.join(ckpt_dir, 'replay_buffer.npz')
    state_path = os.path.join(ckpt_dir, 'train_state.pkl')

    required_paths = [params_path, state_path]
    if load_replay_buffer:
        required_paths.append(buf_path)
    if not all(os.path.exists(p) for p in required_paths):
        print(f"  Incomplete checkpoint in {ckpt_dir}, starting fresh")
        return 0, 0

    # --- Load params ---
    _patch_flax_variablestate_unpickle()
    with open(params_path, 'rb') as f:
        params = pickle.load(f)

    # Restore networks
    nnx.update(agent.policy, _to_jax_tree(params['policy']))
    nnx.update(agent.critic, _to_jax_tree(params['critic']))
    nnx.update(agent.target_critic, _to_jax_tree(params['target_critic']))
    agent.log_alpha = jnp.array(params['log_alpha'])
    agent.update_count = params['update_count']

    # Restore optimizer states
    agent.policy_opt_state = _to_jax_tree(params['policy_opt_state'])
    agent.critic_opt_state = _to_jax_tree(params['critic_opt_state'])
    agent.alpha_opt_state = _to_jax_tree(params['alpha_opt_state'])

    # Context network
    if hasattr(agent, 'context_net') and 'context_net' in params:
        nnx.update(agent.context_net, _to_jax_tree(params['context_net']))
        agent.context_opt_state = _to_jax_tree(params['context_opt_state'])

    # Belief tracker
    if hasattr(agent, 'belief_tracker') and 'belief_state' in params:
        bt_state = params['belief_state']
        agent.belief_tracker.belief = bt_state['belief']
        if 'lw_baseline' in params:
            agent._lw_baseline = params['lw_baseline']
        if 'current_weighted_lambda' in params:
            agent._current_weighted_lambda = params['current_weighted_lambda']
        if 'reg_latched' in params:
            agent._reg_latched = bool(params['reg_latched'])
        if 'bapr_controller_state' in params:
            state = params['bapr_controller_state']
            agent._train_reward_ema = state.get('train_reward_ema')
            agent._train_reward_peak = state.get('train_reward_peak')
            agent._eval_reward_ema = state.get('eval_reward_ema')
            agent._eval_reward_peak = state.get('eval_reward_peak')
            agent._last_train_reward_drop_frac = state.get(
                'last_train_reward_drop_frac', 0.0)
            agent._last_eval_reward_drop_frac = state.get(
                'last_eval_reward_drop_frac', 0.0)
            agent._last_perf_drop_frac = state.get('last_perf_drop_frac', 0.0)
            agent._controller_active = state.get('controller_active', False)
            agent._controller_latched = state.get('controller_latched', False)
            agent._controller_active_iters = state.get(
                'controller_active_iters', 0)
            agent._controller_duration_released = state.get(
                'controller_duration_released', False)
            agent._controller_cooldown_iters = state.get(
                'controller_cooldown_iters', 0)
            agent._controller_exit_reason = state.get(
                'controller_exit_reason', 0)
            agent._controller_signal = state.get('controller_signal', 0.0)
            agent._controller_drawdown = state.get('controller_drawdown', 0.0)
            agent._controller_low_disagreement = state.get(
                'controller_low_disagreement', False)
            agent._last_train_reward_improve_frac = state.get(
                'last_train_reward_improve_frac', 0.0)
            agent._last_eval_reward_improve_frac = state.get(
                'last_eval_reward_improve_frac', 0.0)
            agent._controller_reg_multiplier = state.get(
                'controller_reg_multiplier', 1.0)
            agent._controller_recent_multiplier = state.get(
                'controller_recent_multiplier', 1.0)
            agent._controller_recent_add_frac = state.get(
                'controller_recent_add_frac', 0.0)
            agent._controller_lcb_multiplier = state.get(
                'controller_lcb_multiplier', 1.0)
            agent._controller_actor_update_multiplier = state.get(
                'controller_actor_update_multiplier', 1.0)

    # EMA policy
    if hasattr(agent, 'ema_policy') and 'ema_policy' in params:
        nnx.update(agent.ema_policy, _to_jax_tree(params['ema_policy']))

    # Performance gating
    if hasattr(agent, '_eval_history') and 'eval_history' in params:
        from collections import deque
        agent._eval_history = deque(params['eval_history'],
                                    maxlen=agent._eval_history.maxlen)
        agent._perf_gate_active = params.get('perf_gate_active', False)

    # Pseudo-label
    if hasattr(agent, '_current_pseudo_label') and 'pseudo_label' in params:
        agent._current_pseudo_label = params['pseudo_label']
        agent._last_lambda_was_high = params['last_lambda_was_high']

    # --- Load replay buffer ---
    if load_replay_buffer:
        buf = np.load(buf_path)
        replay_buffer.from_numpy(buf)

    # --- Load training state ---
    with open(state_path, 'rb') as f:
        train_state = pickle.load(f)

    start_iter = train_state['iteration'] + 1  # resume from NEXT iteration
    total_steps = train_state['total_steps']

    # Restore logger data
    from collections import defaultdict
    logger.data = defaultdict(list, train_state['logger_data'])

    print(f"  Checkpoint loaded: resuming from iter={start_iter}, "
          f"steps={total_steps}, buf={replay_buffer.size}")

    return start_iter, total_steps


def has_checkpoint(ckpt_dir: str, require_replay_buffer: bool = True) -> bool:
    """Check if a valid checkpoint exists."""
    required = [
        os.path.join(ckpt_dir, 'params.pkl'),
        os.path.join(ckpt_dir, 'train_state.pkl'),
    ]
    if require_replay_buffer:
        required.append(os.path.join(ckpt_dir, 'replay_buffer.npz'))
    return all(os.path.exists(p) for p in required)


def get_checkpoint_iteration(ckpt_dir: str) -> int:
    """Get the iteration number from a checkpoint. Returns -1 if no checkpoint."""
    state_path = os.path.join(ckpt_dir, 'train_state.pkl')
    if not os.path.exists(state_path):
        return -1
    try:
        with open(state_path, 'rb') as f:
            train_state = pickle.load(f)
        return train_state['iteration']
    except Exception:
        return -1
