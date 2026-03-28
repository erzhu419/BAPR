"""Vectorized ensemble Q-network using independent critics."""
import jax
import jax.numpy as jnp
from flax import nnx


class SingleCritic(nnx.Module):
    """Single Q(s,a) network."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256,
                 n_layers: int = 3, *, rngs: nnx.Rngs):
        layers = []
        in_d = obs_dim + act_dim
        for _ in range(n_layers):
            layers.append(nnx.Linear(in_d, hidden_dim, rngs=rngs))
            in_d = hidden_dim
        layers.append(nnx.Linear(in_d, 1, rngs=rngs))
        self.layers = nnx.List(layers)
        self.n_hidden = n_layers

    def __call__(self, obs, act):
        x = jnp.concatenate([obs, act], axis=-1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.n_hidden:
                x = nnx.relu(x)
        return x.squeeze(-1)


class EnsembleCritic(nnx.Module):
    """Ensemble of N independent Q-networks.

    Forward pass returns [N, batch] Q-values.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256,
                 ensemble_size: int = 10, n_layers: int = 3, *, rngs: nnx.Rngs):
        critics = [
            SingleCritic(obs_dim, act_dim, hidden_dim, n_layers,
                         rngs=nnx.Rngs(params=rngs.params()))
            for _ in range(ensemble_size)
        ]
        self.critics = nnx.List(critics)
        self.ensemble_size = ensemble_size

    def __call__(self, obs, act):
        """Returns [ensemble_size, batch] Q-values."""
        qs = jnp.stack([c(obs, act) for c in self.critics], axis=0)
        return qs

    def compute_reg_norm(self):
        """L1 regularization norm per critic member. Returns [ensemble_size]."""
        norms = []
        for critic in self.critics:
            total = 0.0
            for layer in critic.layers:
                total = total + jnp.sum(jnp.abs(layer.kernel))
                if layer.bias is not None:
                    total = total + jnp.sum(jnp.abs(layer.bias))
            norms.append(total)
        return jnp.array(norms)
