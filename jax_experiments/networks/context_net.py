"""Context Network (Environment Probe) for ESCP/BAPR.

Maps state observations to a low-dimensional context embedding
that identifies the current environment mode.
"""
import jax
import jax.numpy as jnp
from flax import nnx


class ContextNetwork(nnx.Module):
    """MLP context encoder with L2-normalized output.

    Inspired by ESCP's Environment Probe (EP).
    """

    def __init__(self, obs_dim: int, ep_dim: int = 2,
                 hidden_dim: int = 128, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(obs_dim, hidden_dim, rngs=rngs)
        self.ln1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.fc3 = nnx.Linear(hidden_dim, ep_dim, rngs=rngs)
        self.ep_dim = ep_dim

    def __call__(self, obs):
        """Returns L2-normalized context vector [batch, ep_dim]."""
        x = self.fc1(obs)
        x = self.ln1(x)
        x = nnx.leaky_relu(x)
        x = self.fc2(x)
        x = nnx.leaky_relu(x)
        x = self.fc3(x)
        # L2 normalize
        norm = jnp.sqrt(jnp.sum(x ** 2, axis=-1, keepdims=True) + 1e-8)
        return x / norm


def compute_rmdm_loss(ep_tensor, task_ids, rbf_radius=3000.0,
                      consistency_weight=50.0, diversity_weight=0.025):
    """RMDM representation loss: within-task consistency + cross-task diversity.

    Fully JIT-compatible: no boolean indexing, no data-dependent shapes.

    Args:
        ep_tensor: [batch, ep_dim] context embeddings
        task_ids: [batch] integer task IDs
        rbf_radius: RBF kernel bandwidth
        consistency_weight: weight for consistency loss
        diversity_weight: weight for diversity (DPP) loss

    Returns:
        scalar loss
    """
    max_tasks = 20  # static upper bound for number of unique tasks

    # Get unique task IDs with fixed output size (padded with -1)
    unique_tasks = jnp.unique(task_ids, size=max_tasks, fill_value=-1)
    valid_mask = unique_tasks >= 0  # [max_tasks] bool — but used only as float mask

    def _compute_task_stats(task_id):
        """Per-task mean and consistency. Safe for invalid task_id=-1."""
        mask = (task_ids == task_id).astype(jnp.float32)  # [batch]
        count = jnp.maximum(mask.sum(), 1.0)
        mean_ep = jnp.sum(ep_tensor * mask[:, None], axis=0) / count
        diff = ep_tensor - mean_ep[None, :]
        var_ep = jnp.sum(diff ** 2 * mask[:, None], axis=0) / count
        consistency = jnp.sqrt(var_ep + 1e-8).mean()
        return mean_ep, consistency

    # Vectorize over all max_tasks slots (vmap handles invalid ones too)
    all_means, all_cons = jax.vmap(_compute_task_stats)(unique_tasks)
    # all_means: [max_tasks, ep_dim], all_cons: [max_tasks]

    n_valid = valid_mask.astype(jnp.float32).sum()

    # Consistency loss: average over valid tasks only
    valid_f = valid_mask.astype(jnp.float32)  # [max_tasks]
    cons_loss = jnp.sum(all_cons * valid_f) / jnp.maximum(n_valid, 1.0)

    # DPP diversity: RBF kernel over valid task means
    diff = all_means[:, None, :] - all_means[None, :, :]  # [T, T, ep_dim]
    K = jnp.exp(-(diff ** 2).sum(-1) * rbf_radius)  # [T, T]
    K = K + jnp.eye(max_tasks) * 1e-3

    # Mask invalid rows/cols: set K[i,j] = delta_ij for invalid tasks
    outer_valid = valid_f[:, None] * valid_f[None, :]  # [T, T]
    K = K * outer_valid + jnp.eye(max_tasks) * (1.0 - outer_valid)

    div_loss = -jnp.linalg.slogdet(K)[1]

    # Return 0 if fewer than 2 valid tasks
    total_loss = consistency_weight * cons_loss + diversity_weight * div_loss
    return jnp.where(n_valid >= 2, total_loss, 0.0)

