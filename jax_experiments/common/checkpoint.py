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


def load_checkpoint(ckpt_dir: str, agent, replay_buffer, logger, algo: str):
    """Load a training checkpoint. Returns (start_iteration, total_steps).

    Args:
        ckpt_dir: Directory containing checkpoint files.
        agent: Algorithm instance (already constructed with correct architecture).
        replay_buffer: Empty ReplayBuffer instance.
        logger: Logger instance.
        algo: Algorithm name string.

    Returns:
        (start_iteration, total_steps) to resume from.
    """
    params_path = os.path.join(ckpt_dir, 'params.pkl')
    buf_path = os.path.join(ckpt_dir, 'replay_buffer.npz')
    state_path = os.path.join(ckpt_dir, 'train_state.pkl')

    if not all(os.path.exists(p) for p in [params_path, buf_path, state_path]):
        print(f"  Incomplete checkpoint in {ckpt_dir}, starting fresh")
        return 0, 0

    # --- Load params ---
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


def has_checkpoint(ckpt_dir: str) -> bool:
    """Check if a valid checkpoint exists."""
    return (os.path.exists(os.path.join(ckpt_dir, 'params.pkl')) and
            os.path.exists(os.path.join(ckpt_dir, 'replay_buffer.npz')) and
            os.path.exists(os.path.join(ckpt_dir, 'train_state.pkl')))


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
