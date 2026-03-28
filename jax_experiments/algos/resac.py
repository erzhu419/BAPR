"""RE-SAC: SAC + ensemble + OOD reg + LCB policy — scan-fused."""
import jax
import jax.numpy as jnp
import optax
from flax import nnx
import numpy as np

from jax_experiments.algos.sac_base import SACBase


class RESAC(SACBase):
    """RE-SAC with LCB-based policy loss. Overrides scan body."""

    def _build_scan_fn(self):
        gamma = self.config.gamma
        tau = self.config.tau
        beta = self.config.beta
        weight_reg = self.config.weight_reg
        auto_alpha = self.config.auto_alpha
        target_entropy = jnp.array(self.target_entropy)

        gd_policy = nnx.graphdef(self.policy)
        gd_critic = nnx.graphdef(self.critic)
        gd_target = nnx.graphdef(self.target_critic)

        p_opt = self.policy_opt
        c_opt = self.critic_opt
        a_opt = self.alpha_opt

        @jax.jit
        def _scan_update(critic_params, target_params, policy_params,
                         log_alpha, c_opt_state, p_opt_state, a_opt_state,
                         all_obs, all_act, all_rew, all_next_obs, all_done,
                         rng_key):

            def body_fn(carry, batch_data):
                (c_p, t_p, p_p, la, c_os, p_os, a_os, key) = carry
                (obs, act, rew, next_obs, done) = batch_data
                key, k1, k2 = jax.random.split(key, 3)
                alpha = jnp.exp(la)

                # === Critic ===
                def critic_loss_fn(cp):
                    tm = nnx.merge(gd_target, t_p)
                    pm = nnx.merge(gd_policy, p_p)
                    na, nlp = pm.sample(next_obs, k1)
                    tq = tm(next_obs, na).min(axis=0) - alpha * nlp
                    tv = rew.squeeze(-1) + gamma * (1 - done.squeeze(-1)) * tq
                    cm = nnx.merge(gd_critic, cp)
                    pq = cm(obs, act)
                    return jnp.mean((pq - tv[None]) ** 2), pq

                (c_loss, pq), c_grads = jax.value_and_grad(
                    critic_loss_fn, has_aux=True)(c_p)
                c_upd, new_c_os = c_opt.update(c_grads, c_os, c_p)
                new_c_p = optax.apply_updates(c_p, c_upd)

                # === Policy (LCB + weight_reg) ===
                def policy_loss_fn(pp):
                    pm = nnx.merge(gd_policy, pp)
                    cm = nnx.merge(gd_critic, new_c_p)
                    na, lp = pm.sample(obs, k2)
                    qv = cm(obs, na)
                    qm = qv.mean(axis=0)
                    qs = qv.std(axis=0)
                    lcb = qm + beta * qs
                    reg = cm.compute_reg_norm()
                    return (jnp.exp(la) * lp - lcb).mean() + weight_reg * reg.mean(), lp

                (p_loss, lp), p_grads = jax.value_and_grad(
                    policy_loss_fn, has_aux=True)(p_p)
                p_upd, new_p_os = p_opt.update(p_grads, p_os, p_p)
                new_p_p = optax.apply_updates(p_p, p_upd)

                # === Alpha ===
                a_grad = -(lp.mean() + target_entropy)
                a_upd, new_a_os = a_opt.update(a_grad, a_os, la)
                new_la = jnp.where(auto_alpha, la + a_upd, la)
                new_a_os = jax.tree.map(
                    lambda n, o: jnp.where(auto_alpha, n, o), new_a_os, a_os)

                # === Target ===
                new_t_p = jax.tree.map(
                    lambda tp, cp: tp * (1 - tau) + cp * tau, t_p, new_c_p)

                new_carry = (new_c_p, new_t_p, new_p_p, new_la,
                             new_c_os, new_p_os, new_a_os, key)
                metrics = (c_loss, p_loss, jnp.exp(new_la),
                           pq.mean(), pq.std(axis=0).mean(), lp.mean())
                return new_carry, metrics

            init = (critic_params, target_params, policy_params,
                    log_alpha, c_opt_state, p_opt_state, a_opt_state, rng_key)
            return jax.lax.scan(body_fn, init,
                               (all_obs, all_act, all_rew, all_next_obs, all_done))

        self._scan_update = _scan_update
