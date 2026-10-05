"""FP32 JAX port of RE-SAC/experiment's Torch networks and update equations."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from jax_experiments.algos.resac import reference_reg_norm
from jax_experiments.algos.sac_base import SACBase
from jax_experiments.networks.ensemble_critic import EnsembleCritic
from jax_experiments.networks.policy import GaussianPolicy


class TorchPolicy(GaussianPolicy):
    def __init__(self, obs_dim, act_dim, hidden_dim, *, rngs):
        super().__init__(obs_dim, act_dim, hidden_dim, rngs=rngs)
        for layer in self.layers:
            # rlkit.fanin_init uses weight.shape[0], i.e. Torch's output axis.
            bound = 1.0 / jnp.sqrt(float(layer.kernel.value.shape[1]))
            layer.kernel.value = jax.random.uniform(rngs.params(), layer.kernel.value.shape,
                                                     minval=-bound, maxval=bound)
            layer.bias.value = jnp.full_like(layer.bias.value, 0.1)
        for layer in (self.mean_head, self.log_std_head):
            layer.kernel.value = jax.random.uniform(rngs.params(), layer.kernel.value.shape,
                                                     minval=-0.001, maxval=0.001)
            layer.bias.value = jax.random.uniform(rngs.params(), layer.bias.value.shape,
                                                   minval=-0.001, maxval=0.001)

    def sample_with_noise(self, obs, noise):
        mean, log_std = self(obs)
        std = jnp.exp(log_std)
        pre_tanh = mean + std * noise
        action = jnp.tanh(pre_tanh)
        log_prob = -0.5 * ((pre_tanh - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)
        log_prob -= jnp.log(1 - action * action + 1e-6)
        return action, log_prob.sum(axis=-1)

    def sample(self, obs, key, ep_tensor=None):
        return self.sample_with_noise(obs, jax.random.normal(key, (*obs.shape[:-1], self.act_dim)))


class TorchCritic(EnsembleCritic):
    def __init__(self, obs_dim, act_dim, hidden_dim, ensemble_size, *, rngs):
        super().__init__(obs_dim, act_dim, hidden_dim, ensemble_size, rngs=rngs)
        self.norm = nnx.LayerNorm(hidden_dim, epsilon=1e-5, use_fast_variance=False, rngs=rngs)
        for layer in self.layers:
            # Original vectorized weight is [in,out]; Torch therefore treats out as fan_in.
            bound = 1.0 / jnp.sqrt(float(layer.kernel.value.shape[-1]))
            layer.kernel.value = jax.random.uniform(rngs.params(), layer.kernel.value.shape,
                                                     minval=-bound, maxval=bound)
            layer.bias.value = jax.random.uniform(rngs.params(), layer.bias.value.shape,
                                                   minval=-bound, maxval=bound)

    def __call__(self, obs, act):
        x = jnp.concatenate((obs, act), axis=-1)
        x = jnp.broadcast_to(x[None], (self.ensemble_size, *x.shape))
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index == 0:
                x = self.norm(x)
            if index < self.n_hidden:
                x = jax.nn.relu(x)
        return x.squeeze(-1)


@jax.custom_jvp
def _torch_sqrt(value):
    return jnp.sqrt(value)


@_torch_sqrt.defjvp
def _torch_sqrt_jvp(primals, tangents):
    (value,), (tangent,) = primals, tangents
    root = _torch_sqrt(value)
    derivative = jnp.where(root > 0, 0.5 / jnp.where(root > 0, root, 1.0), 0.0)
    return root, derivative * tangent


def sample_std(values):
    """Torch's unbiased ensemble std, including its zero-variance derivative."""
    centered = values - values.mean(axis=0)
    return _torch_sqrt((centered * centered).sum(axis=0) / (values.shape[0] - 1))


def bus_reward(reward):
    """Original bus-core transform for one minibatch, never the update-stack axis."""
    std = sample_std(reward)
    std = jnp.where(std < 1e-6, jnp.ones_like(std), std)
    return 10 * (reward - reward.mean(axis=0)) / std


class TorchEquationRESAC(SACBase):
    def __init__(self, obs_dim, act_dim, config, seed=0):
        self.config, self.obs_dim, self.act_dim = config, obs_dim, act_dim
        self.rngs = nnx.Rngs(seed)
        self.policy = TorchPolicy(obs_dim, act_dim, config.hidden_dim, rngs=self.rngs)
        self.critic = TorchCritic(obs_dim, act_dim, config.hidden_dim,
                                 config.ensemble_size, rngs=self.rngs)
        # NNX graph cloning avoids legacy pickle hooks changing Param metadata.
        self.target_critic = nnx.merge(*nnx.split(self.critic))
        self.log_alpha = jnp.array([0.0]) if config.auto_alpha else jnp.array([jnp.log(config.alpha)])
        self.target_entropy = -float(act_dim)
        self.policy_opt = optax.adam(config.lr)
        self.critic_opt = optax.adam(config.lr)
        self.alpha_opt = optax.adam(config.lr)
        self.policy_opt_state = self.policy_opt.init(nnx.state(self.policy, nnx.Param))
        self.critic_opt_state = self.critic_opt.init(nnx.state(self.critic, nnx.Param))
        self.alpha_opt_state = self.alpha_opt.init(self.log_alpha)
        self.update_count = 0
        self._build_scan_fn()

    def checkpoint_state(self):
        return {"rng_leaves": [
            np.asarray(jax.random.key_data(value))
            if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key) else np.asarray(value)
            for value in jax.tree.leaves(nnx.state(self.rngs))]}

    def load_checkpoint_state(self, state):
        leaves, tree = jax.tree.flatten(nnx.state(self.rngs))
        restored = [jax.random.wrap_key_data(jnp.asarray(saved))
                    if jax.dtypes.issubdtype(template.dtype, jax.dtypes.prng_key) else jnp.asarray(saved)
                    for template, saved in zip(leaves, state["rng_leaves"], strict=True)]
        nnx.update(self.rngs, jax.tree.unflatten(tree, restored))

    def _build_scan_fn(self):
        config = self.config
        policy_graph, critic_graph = nnx.graphdef(self.policy), nnx.graphdef(self.critic)
        target_graph = nnx.graphdef(self.target_critic)

        def clip(grads):
            if config.resac_clip_norm <= 0:
                return grads
            factor = jnp.minimum(1.0, config.resac_clip_norm / (optax.global_norm(grads) + 1e-6))
            return jax.tree.map(lambda grad: grad * factor, grads)

        @jax.jit
        def scan_update(critic_params, target_params, policy_params, log_alpha,
                        critic_state, policy_state, alpha_state, batch,
                        policy_noise, next_noise, update_offset):
            def step(carry, data):
                cp, tp, pp, la, cos, pos, aos = carry
                obs, act, rew, next_obs, done, noise, target_noise, index = data
                training_rew = bus_reward(rew) if config.resac_reward_transform == "bus_standardized" else rew
                policy = nnx.merge(policy_graph, pp)
                _, log_prob = policy.sample_with_noise(obs, noise)
                if config.auto_alpha:
                    alpha_grad = -jnp.exp(la) * jax.lax.stop_gradient(log_prob.mean() + self.target_entropy)
                    delta, aos = self.alpha_opt.update(alpha_grad, aos, la)
                    la = optax.apply_updates(la, delta)
                alpha = jax.lax.stop_gradient(jnp.exp(la))
                target = nnx.merge(target_graph, tp)
                bonus = config.weight_reg * reference_reg_norm(target)[:, None]
                next_action, next_log_prob = policy.sample_with_noise(next_obs, target_noise)
                bootstrap = target(next_obs, next_action) + bonus - alpha * next_log_prob
                q_target = training_rew.squeeze(-1) + config.gamma * (1 - done.squeeze(-1)) * bootstrap

                def critic_loss(candidate):
                    q = nnx.merge(critic_graph, candidate)(obs, act)
                    loss = ((q - q_target) ** 2).mean() + config.beta_ood * sample_std(q).mean()
                    return loss, q

                (critic_loss_value, q), grads = jax.value_and_grad(critic_loss, has_aux=True)(cp)
                delta, cos = self.critic_opt.update(clip(grads), cos, cp)
                cp = optax.apply_updates(cp, delta)

                def update_actor(operand):
                    current, state = operand

                    def policy_loss(candidate):
                        action, lp = nnx.merge(policy_graph, candidate).sample_with_noise(obs, noise)
                        values = nnx.merge(critic_graph, cp)(obs, action) + bonus - alpha * lp
                        bc = ((action - act) ** 2).mean()
                        loss = -(values.mean(axis=0) + config.beta * sample_std(values)).mean()
                        return loss + config.resac_beta_bc * bc, bc

                    (loss, bc), grads = jax.value_and_grad(policy_loss, has_aux=True)(current)
                    delta, state = self.policy_opt.update(clip(grads), state, current)
                    return optax.apply_updates(current, delta), state, loss, bc

                actor_updated = index % config.resac_critic_actor_ratio == 0
                pp, pos, policy_loss_value, bc = jax.lax.cond(
                    actor_updated, update_actor,
                    lambda operand: (*operand, jnp.array(0.0), jnp.array(0.0)), (pp, pos))
                tp = jax.tree.map(lambda old, new: (1 - config.tau) * old + config.tau * new, tp, cp)
                metrics = (critic_loss_value, policy_loss_value, alpha.mean(), q.mean(),
                           sample_std(q).mean(), log_prob.mean(), bc,
                           actor_updated.astype(jnp.float32), bonus.mean(), bonus.std(),
                           rew.mean(), sample_std(rew).mean(), training_rew.mean(), sample_std(training_rew).mean())
                return (cp, tp, pp, la, cos, pos, aos), metrics

            data = (batch["obs"], batch["act"], batch["rew"], batch["next_obs"], batch["done"],
                    policy_noise, next_noise, jnp.arange(batch["obs"].shape[0]) + update_offset)
            return jax.lax.scan(step, (critic_params, target_params, policy_params, log_alpha,
                                       critic_state, policy_state, alpha_state), data)

        self._scan_update = scan_update

    def multi_update(self, stacked_batch, **kwargs):
        del kwargs
        keys = jax.random.split(self.rngs.params())
        shape = (*stacked_batch["obs"].shape[:-1], self.act_dim)
        noise, next_noise = (jax.random.normal(key, shape) for key in keys)
        final, metrics = self._scan_update(
            nnx.state(self.critic, nnx.Param), nnx.state(self.target_critic, nnx.Param),
            nnx.state(self.policy, nnx.Param), self.log_alpha,
            self.critic_opt_state, self.policy_opt_state, self.alpha_opt_state,
            stacked_batch, noise, next_noise, self.update_count)
        cp, tp, pp, self.log_alpha, self.critic_opt_state, self.policy_opt_state, self.alpha_opt_state = final
        nnx.update(self.critic, cp)
        nnx.update(self.target_critic, tp)
        nnx.update(self.policy, pp)
        self.update_count += stacked_batch["obs"].shape[0]
        names = ("critic_loss", "policy_loss", "alpha", "q_mean", "q_std_mean", "log_prob",
                 "bc_loss", "actor_update_rate", "reg_bonus_mean", "reg_bonus_std",
                 "raw_reward_mean", "raw_reward_std", "training_reward_mean", "training_reward_std")
        result = {name: float(value.mean()) for name, value in zip(names, metrics)}
        result["alpha"] = float(self.alpha.mean())
        return result
