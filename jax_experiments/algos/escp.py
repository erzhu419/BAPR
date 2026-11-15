"""ESCP-style context-conditioned SAC with legacy and paper-core modes."""
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from jax_experiments.networks.context_net import (
    ContextNetwork,
    compute_rmdm_loss,
)
from jax_experiments.networks.ensemble_critic import EnsembleCritic
from jax_experiments.networks.escp_recurrent import (
    ESCPGaussianPolicy,
    ESCPTwinCritic,
    RecurrentEnvironmentProbe,
)
from jax_experiments.networks.policy import GaussianPolicy


class ESCP:
    """Context-conditioned SAC with an RMDM representation objective.

    The legacy BAPR fork used independent critic targets and an ensemble LCB
    actor.  The original ESCP controller instead used the minimum of two Q
    functions for both targets and actor updates.  Both modes remain explicit
    so old checkpoints resume with their original semantics.
    """

    def __init__(self, obs_dim: int, act_dim: int, config, seed: int = 0):
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rngs = nnx.Rngs(seed)
        self.context_mode = str(getattr(
            config, "escp_context_mode", "state_mlp"))
        if self.context_mode not in ("state_mlp", "recurrent"):
            raise ValueError(
                f"unsupported ESCP context mode {self.context_mode!r}")
        self.uses_recurrent_context = self.context_mode == "recurrent"

        if self.uses_recurrent_context:
            self.context_net = RecurrentEnvironmentProbe(
                obs_dim, act_dim, config.ep_dim, rngs=self.rngs)
            self.policy = ESCPGaussianPolicy(
                obs_dim, act_dim, config.ep_dim, rngs=self.rngs)
            self.critic = ESCPTwinCritic(
                obs_dim + config.ep_dim, act_dim,
                ensemble_size=config.ensemble_size, rngs=self.rngs)
        else:
            self.context_net = ContextNetwork(
                obs_dim, config.ep_dim, hidden_dim=128, rngs=self.rngs)
            self.policy = GaussianPolicy(
                obs_dim, act_dim, config.hidden_dim,
                ep_dim=config.ep_dim, n_layers=2, rngs=self.rngs)
            self.critic = EnsembleCritic(
                obs_dim + config.ep_dim, act_dim, config.hidden_dim,
                ensemble_size=config.ensemble_size, n_layers=3,
                rngs=self.rngs)
        self.target_critic = deepcopy(self.critic)

        self.log_alpha = jnp.array(jnp.log(config.alpha))
        entropy_ratio = (
            float(getattr(config, "escp_target_entropy_ratio", 1.0))
            if self.uses_recurrent_context else 1.0)
        self.target_entropy = -entropy_ratio * float(act_dim)

        def selected_lr(name):
            value = float(getattr(config, name, -1.0))
            return value if value > 0.0 else float(config.lr)

        self.policy_opt = optax.adam(selected_lr("escp_policy_lr"))
        self.critic_opt = optax.adam(selected_lr("escp_critic_lr"))
        self.context_opt = optax.adam(selected_lr("escp_context_lr"))
        self.alpha_opt = optax.adam(selected_lr("escp_alpha_lr"))

        self.policy_opt_state = self.policy_opt.init(
            nnx.state(self.policy, nnx.Param))
        self.critic_opt_state = self.critic_opt.init(
            nnx.state(self.critic, nnx.Param))
        self.context_opt_state = self.context_opt.init(
            nnx.state(self.context_net, nnx.Param))
        self.alpha_opt_state = self.alpha_opt.init(self.log_alpha)

        max_tasks = int(getattr(config, "rmdm_max_tasks", 64))
        if self.uses_recurrent_context and int(config.task_num) > max_tasks:
            raise ValueError(
                "recurrent ESCP requires rmdm_max_tasks >= task_num")
        self.context_prototypes = jnp.zeros(
            (max_tasks, int(config.ep_dim)), dtype=jnp.float32)
        self.context_prototype_valid = jnp.zeros(
            (max_tasks,), dtype=jnp.bool_)
        self.reset_recurrent_context()

        self.update_count = 0
        self._build_scan_fn()

    def _build_scan_fn(self):
        if self.uses_recurrent_context:
            self._build_recurrent_scan_fn()
            return
        gamma = self.config.gamma
        tau = self.config.tau
        beta = self.config.beta
        auto_alpha = self.config.auto_alpha
        target_entropy = jnp.array(self.target_entropy)
        rbf_r = self.config.rbf_radius
        cons_w = self.config.consistency_loss_weight
        div_w = self.config.diversity_loss_weight
        rmdm_max_tasks = int(getattr(self.config, "rmdm_max_tasks", 64))
        target_mode = str(getattr(
            self.config, "escp_target_mode", "independent"))
        actor_mode = str(getattr(self.config, "escp_actor_mode", "lcb"))
        if target_mode not in ("independent", "twin_min"):
            raise ValueError(f"unsupported ESCP target mode {target_mode!r}")
        if actor_mode not in ("lcb", "twin_min"):
            raise ValueError(f"unsupported ESCP actor mode {actor_mode!r}")
        paper_core = target_mode == "twin_min" and actor_mode == "twin_min"
        clip_norm = float(self.config.clip_norm) if paper_core else 0.0
        alpha_max = float(getattr(self.config, "escp_alpha_max", -1.0))

        gd_policy = nnx.graphdef(self.policy)
        gd_critic = nnx.graphdef(self.critic)
        gd_target = nnx.graphdef(self.target_critic)
        gd_ctx = nnx.graphdef(self.context_net)

        p_opt = self.policy_opt
        c_opt = self.critic_opt
        x_opt = self.context_opt
        a_opt = self.alpha_opt

        def clip_grads(grads):
            if clip_norm <= 0.0:
                return grads
            grad_norm = optax.global_norm(grads)
            scale = jnp.minimum(
                1.0, jnp.asarray(clip_norm) / (grad_norm + 1e-8))
            return jax.tree.map(lambda grad: grad * scale, grads)

        def tree_all_finite(tree):
            return jnp.all(jnp.stack([
                jnp.all(jnp.isfinite(leaf))
                for leaf in jax.tree.leaves(tree)
            ]))

        @jax.jit
        def _scan_update(critic_params, target_params, policy_params,
                         ctx_params, log_alpha,
                         c_opt_state, p_opt_state, x_opt_state, a_opt_state,
                         all_obs, all_act, all_rew, all_next_obs, all_done,
                         all_task_ids, rng_key, context_active,
                         context_train):

            def body_fn(carry, batch_data):
                (c_p, t_p, p_p, x_p, la, c_os, p_os, x_os, a_os,
                 key) = carry
                (obs, act, rew, next_obs, done, tids) = batch_data
                key, k1, k2 = jax.random.split(key, 3)
                alpha = jnp.exp(la)

                def ctx_loss_fn(xp):
                    xm = nnx.merge(gd_ctx, xp)
                    ep = xm(obs)
                    return compute_rmdm_loss(
                        ep, tids, rbf_r, cons_w, div_w, rmdm_max_tasks)

                x_loss, x_grads = jax.value_and_grad(ctx_loss_fn)(x_p)
                x_grad_norm = optax.global_norm(x_grads)
                x_grads = clip_grads(x_grads)
                x_upd, candidate_x_os = x_opt.update(x_grads, x_os, x_p)
                candidate_x_p = optax.apply_updates(x_p, x_upd)
                new_x_p = jax.tree.map(
                    lambda new, old: jnp.where(context_train, new, old),
                    candidate_x_p, x_p)
                new_x_os = jax.tree.map(
                    lambda new, old: jnp.where(context_train, new, old),
                    candidate_x_os, x_os)

                ctx_model = nnx.merge(gd_ctx, new_x_p)
                ep = ctx_model(obs)
                ep = jnp.where(context_active, ep, jnp.zeros_like(ep))
                next_ep = ctx_model(next_obs)
                next_ep = jnp.where(
                    context_active, next_ep, jnp.zeros_like(next_ep))
                obs_aug = jnp.concatenate([obs, ep], axis=-1)
                next_aug = jnp.concatenate([next_obs, next_ep], axis=-1)

                def critic_loss_fn(cp):
                    tm = nnx.merge(gd_target, t_p)
                    pm = nnx.merge(gd_policy, p_p)
                    next_action, next_log_prob = pm.sample(
                        next_obs, k1, next_ep)
                    target_q_all = tm(next_aug, next_action)
                    if target_mode == "twin_min":
                        bootstrap = (
                            target_q_all.min(axis=0)
                            - alpha * next_log_prob)
                        target = (
                            rew.squeeze(-1)
                            + gamma * (1 - done.squeeze(-1)) * bootstrap)
                    else:
                        bootstrap = target_q_all - alpha * next_log_prob
                        target = (
                            rew.squeeze(-1)
                            + gamma * (1 - done.squeeze(-1)) * bootstrap)
                    cm = nnx.merge(gd_critic, cp)
                    predicted_q = cm(obs_aug, act)
                    if target_mode == "twin_min":
                        loss = jnp.mean(
                            (predicted_q - target[None]) ** 2)
                    else:
                        loss = jnp.mean((predicted_q - target) ** 2)
                    return loss, predicted_q

                (critic_loss, predicted_q), c_grads = jax.value_and_grad(
                    critic_loss_fn, has_aux=True)(c_p)
                c_grad_norm = optax.global_norm(c_grads)
                c_grads = clip_grads(c_grads)
                c_upd, new_c_os = c_opt.update(c_grads, c_os, c_p)
                new_c_p = optax.apply_updates(c_p, c_upd)

                def policy_loss_fn(pp):
                    pm = nnx.merge(gd_policy, pp)
                    cm = nnx.merge(gd_critic, new_c_p)
                    new_action, log_prob = pm.sample(obs, k2, ep)
                    q_values = cm(obs_aug, new_action)
                    if actor_mode == "twin_min":
                        actor_q = q_values.min(axis=0)
                    else:
                        actor_q = (
                            q_values.mean(axis=0)
                            + beta * q_values.std(axis=0))
                    loss = (jnp.exp(la) * log_prob - actor_q).mean()
                    return loss, log_prob

                (policy_loss, log_prob), p_grads = jax.value_and_grad(
                    policy_loss_fn, has_aux=True)(p_p)
                p_grad_norm = optax.global_norm(p_grads)
                p_grads = clip_grads(p_grads)
                p_upd, new_p_os = p_opt.update(p_grads, p_os, p_p)
                new_p_p = optax.apply_updates(p_p, p_upd)

                if paper_core:
                    alpha_grad = (
                        -jnp.exp(la)
                        * (log_prob.mean() + target_entropy))
                else:
                    alpha_grad = -(log_prob.mean() + target_entropy)
                alpha_upd, candidate_a_os = a_opt.update(
                    alpha_grad, a_os, la)
                new_la = jnp.where(auto_alpha, la + alpha_upd, la)
                if alpha_max > 0.0:
                    new_la = jnp.minimum(new_la, jnp.log(alpha_max))
                new_a_os = jax.tree.map(
                    lambda new, old: jnp.where(auto_alpha, new, old),
                    candidate_a_os, a_os)

                new_t_p = jax.tree.map(
                    lambda target_p, critic_p: (
                        target_p * (1 - tau) + critic_p * tau),
                    t_p, new_c_p)

                scalar_finite = jnp.all(jnp.isfinite(jnp.stack([
                    critic_loss, policy_loss, x_loss, new_la,
                    c_grad_norm, p_grad_norm, x_grad_norm,
                ])))
                parameter_finite = jnp.logical_and(
                    tree_all_finite(new_c_p),
                    jnp.logical_and(
                        tree_all_finite(new_p_p),
                        tree_all_finite(new_x_p)))
                step_finite = jnp.logical_and(
                    scalar_finite, parameter_finite)

                new_carry = (
                    new_c_p, new_t_p, new_p_p, new_x_p, new_la,
                    new_c_os, new_p_os, new_x_os, new_a_os, key)
                metrics = (
                    critic_loss, policy_loss, x_loss, jnp.exp(new_la),
                    predicted_q.mean(),
                    predicted_q.std(axis=0).mean(), log_prob.mean(),
                    c_grad_norm, p_grad_norm, x_grad_norm, step_finite)
                return new_carry, metrics

            init = (
                critic_params, target_params, policy_params, ctx_params,
                log_alpha, c_opt_state, p_opt_state, x_opt_state,
                a_opt_state, rng_key)
            batches = (
                all_obs, all_act, all_rew, all_next_obs, all_done,
                all_task_ids)
            return jax.lax.scan(body_fn, init, batches)

        self._scan_update = _scan_update

    def _build_recurrent_scan_fn(self):
        """Build the released recurrent ESCP update path.

        The environment probe is optimized only by RMDM. Controller actions
        use the causal probe output, while Q functions use a slow per-task
        prototype, matching the released ``stop_pg_for_ep`` implementation.
        """
        target_mode = str(getattr(
            self.config, "escp_target_mode", "independent"))
        actor_mode = str(getattr(
            self.config, "escp_actor_mode", "lcb"))
        if target_mode != "twin_min" or actor_mode != "twin_min":
            raise ValueError(
                "recurrent ESCP requires twin_min targets and actor")

        gamma = float(self.config.gamma)
        tau = float(self.config.tau)
        auto_alpha = bool(self.config.auto_alpha)
        target_entropy = jnp.asarray(self.target_entropy, dtype=jnp.float32)
        rbf_r = float(self.config.rbf_radius)
        cons_w = float(self.config.consistency_loss_weight)
        div_w = float(self.config.diversity_loss_weight)
        max_tasks = int(getattr(self.config, "rmdm_max_tasks", 64))
        prototype_tau = float(getattr(
            self.config, "escp_prototype_tau", 0.995))
        bottleneck_sigma = float(getattr(
            self.config, "escp_bottleneck_sigma", 0.0))
        clip_norm = float(self.config.clip_norm)
        alpha_max = float(getattr(self.config, "escp_alpha_max", -1.0))

        gd_policy = nnx.graphdef(self.policy)
        gd_critic = nnx.graphdef(self.critic)
        gd_target = nnx.graphdef(self.target_critic)
        # Flax 0.10 GRUCell retains two RngState leaves for carry
        # initialization, while newer Flax versions do not.  Split the probe
        # exhaustively so its graph can be merged with trainable parameters on
        # both scheduler and local runtimes.
        gd_ctx, _, ctx_non_params = nnx.split(
            self.context_net, nnx.Param, ...)
        p_opt = self.policy_opt
        c_opt = self.critic_opt
        x_opt = self.context_opt
        a_opt = self.alpha_opt

        def clip_grads(grads):
            if clip_norm <= 0.0:
                return grads
            grad_norm = optax.global_norm(grads)
            scale = jnp.minimum(
                1.0, jnp.asarray(clip_norm) / (grad_norm + 1e-8))
            return jax.tree.map(lambda grad: grad * scale, grads)

        def tree_all_finite(tree):
            return jnp.all(jnp.stack([
                jnp.all(jnp.isfinite(leaf))
                for leaf in jax.tree.leaves(tree)
            ]))

        def update_prototypes(prototypes, valid, contexts, task_ids):
            ids = jnp.arange(max_tasks, dtype=jnp.int32)

            def task_stats(task_id):
                mask = (task_ids == task_id).astype(jnp.float32)
                count = mask.sum()
                mean = jnp.sum(contexts * mask[:, None], axis=0) / jnp.maximum(
                    count, 1.0)
                return mean, count > 0.0

            means, seen = jax.vmap(task_stats)(ids)
            blended = jnp.where(
                valid[:, None],
                prototype_tau * prototypes + (1.0 - prototype_tau) * means,
                means,
            )
            return (
                jnp.where(seen[:, None], blended, prototypes),
                jnp.logical_or(valid, seen),
            )

        def recurrent_rmdm_loss(
                contexts, task_ids, prototypes, prototype_valid):
            """Released timing-RMDM objective without a cuSolver dependency."""
            ids = jnp.arange(max_tasks, dtype=jnp.int32)

            def task_stats(task_id):
                mask = (task_ids == task_id).astype(jnp.float32)
                count = mask.sum()
                mean = jnp.sum(contexts * mask[:, None], axis=0) / jnp.maximum(
                    count, 1.0)
                return mean, count, mask

            means, counts, masks = jax.vmap(task_stats)(ids)
            seen = counts > 0.0
            valid = jnp.logical_or(prototype_valid, seen)
            reference = jnp.where(
                prototype_valid[:, None], prototypes,
                jax.lax.stop_gradient(means))
            squared = jnp.sum(
                (contexts[None, :, :] - reference[:, None, :]) ** 2,
                axis=-1)
            consistency_variance = jnp.sum(
                squared * masks) / jnp.maximum(
                    contexts.shape[0] * contexts.shape[1], 1)
            consistency = jnp.sqrt(
                jnp.maximum(consistency_variance, 1e-12))

            representations = jnp.where(
                seen[:, None], means, prototypes)
            differences = (
                representations[:, None, :]
                - representations[None, :, :])
            # The released default is ``rbf_element_wise``.
            kernel = jnp.exp(
                -rbf_r * differences ** 2).mean(axis=-1)
            valid_pair = valid[:, None] & valid[None, :]
            matrix = jnp.where(valid_pair, kernel, 0.0)
            diagonal = jnp.where(valid, 1.001, 1.0)
            matrix = matrix.at[jnp.diag_indices(max_tasks)].set(diagonal)

            # Static-shape Cholesky avoids the cuSolver availability mismatch
            # that made jnp.linalg.slogdet fail on some scheduler nodes.
            row_ids = jnp.arange(max_tasks)

            def cholesky_row(index, lower):
                prefix = (row_ids < index).astype(matrix.dtype)
                current_prefix = lower[index] * prefix
                diagonal_value = jnp.sqrt(jnp.maximum(
                    matrix[index, index]
                    - jnp.sum(current_prefix ** 2), 1e-8))
                numerator = (
                    matrix[:, index]
                    - jnp.sum(lower * current_prefix[None, :], axis=1))
                column = numerator / diagonal_value
                column = jnp.where(row_ids > index, column, 0.0)
                lower = lower.at[:, index].set(column)
                return lower.at[index, index].set(diagonal_value)

            lower = jax.lax.fori_loop(
                0, max_tasks, cholesky_row, jnp.zeros_like(matrix))
            logdet = 2.0 * jnp.log(jnp.diag(lower)).sum()
            diversity = -logdet
            task_count = valid.astype(jnp.int32).sum()
            objective = cons_w * consistency + jnp.where(
                consistency >= 0.1, 0.0, div_w * diversity)
            return jnp.where(task_count >= 2, objective, 0.0)

        @jax.jit
        def _scan_update(
                critic_params, target_params, policy_params, ctx_params,
                log_alpha, c_opt_state, p_opt_state, x_opt_state,
                a_opt_state, prototypes, prototype_valid,
                all_obs, all_prev_act, all_reset_before, all_act, all_rew,
                all_next_obs, all_next_prev_act, all_next_reset_before,
                all_done, all_task_ids, rng_key, context_active,
                context_train):

            def body_fn(carry, batch_data):
                (c_p, t_p, p_p, x_p, la, c_os, p_os, x_os, a_os,
                 proto, proto_valid, key) = carry
                (obs_history, prev_act_history, reset_before, act, rew,
                 next_obs_history, next_prev_act_history, next_reset_before,
                 done, task_ids) = batch_data
                key, action_key, next_action_key, noise_key, next_noise_key = (
                    jax.random.split(key, 5))
                alpha = jnp.exp(la)

                def context_loss_fn(xp):
                    context_model = nnx.merge(
                        gd_ctx, xp, ctx_non_params)
                    _, sequence_context = context_model.sequence(
                        obs_history, prev_act_history, reset_before)
                    final_context = sequence_context[:, -1, :]
                    return recurrent_rmdm_loss(
                        final_context, task_ids, proto, proto_valid)

                context_loss, context_grads = jax.value_and_grad(
                    context_loss_fn)(x_p)
                context_grad_norm = optax.global_norm(context_grads)
                context_grads = clip_grads(context_grads)
                context_updates, candidate_x_os = x_opt.update(
                    context_grads, x_os, x_p)
                candidate_x_p = optax.apply_updates(x_p, context_updates)
                new_x_p = jax.tree.map(
                    lambda new, old: jnp.where(context_train, new, old),
                    candidate_x_p, x_p)
                new_x_os = jax.tree.map(
                    lambda new, old: jnp.where(context_train, new, old),
                    candidate_x_os, x_os)

                context_model = nnx.merge(
                    gd_ctx, new_x_p, ctx_non_params)
                _, current_sequence = context_model.sequence(
                    obs_history, prev_act_history, reset_before)
                _, next_sequence = context_model.sequence(
                    next_obs_history, next_prev_act_history,
                    next_reset_before)
                inferred_context = current_sequence[:, -1, :]
                inferred_next_context = next_sequence[:, -1, :]

                candidate_proto, candidate_proto_valid = update_prototypes(
                    proto, proto_valid, inferred_context, task_ids)
                # Released timing-RMDM updates history means even during the
                # 100k-step delay that suppresses EP gradient updates.
                new_proto = candidate_proto
                new_proto_valid = candidate_proto_valid
                bounded_ids = jnp.clip(task_ids, 0, max_tasks - 1)
                task_context = jnp.where(
                    new_proto_valid[bounded_ids][:, None],
                    new_proto[bounded_ids], inferred_context)

                inferred_context = jnp.where(
                    context_active, inferred_context,
                    jnp.zeros_like(inferred_context))
                inferred_next_context = jnp.where(
                    context_active, inferred_next_context,
                    jnp.zeros_like(inferred_next_context))
                task_context = jnp.where(
                    context_active, task_context,
                    jnp.zeros_like(task_context))

                if bottleneck_sigma > 0.0:
                    noise_scale = jnp.where(
                        context_active, bottleneck_sigma, 0.0)
                    actor_context = inferred_context + noise_scale * (
                        jax.random.normal(noise_key, inferred_context.shape))
                    next_actor_context = inferred_next_context + noise_scale * (
                        jax.random.normal(
                            next_noise_key, inferred_next_context.shape))
                else:
                    actor_context = inferred_context
                    next_actor_context = inferred_next_context

                obs = obs_history[:, -1, :]
                next_obs = next_obs_history[:, -1, :]
                obs_aug = jnp.concatenate([obs, task_context], axis=-1)
                next_obs_aug = jnp.concatenate(
                    [next_obs, task_context], axis=-1)

                def critic_loss_fn(cp):
                    target_model = nnx.merge(gd_target, t_p)
                    policy_model = nnx.merge(gd_policy, p_p)
                    next_action, next_log_prob = policy_model.sample(
                        next_obs, next_action_key, next_actor_context)
                    target_q = target_model(next_obs_aug, next_action)
                    bootstrap = target_q.min(axis=0) - alpha * next_log_prob
                    target = (
                        rew.squeeze(-1)
                        + gamma * (1.0 - done.squeeze(-1)) * bootstrap)
                    critic_model = nnx.merge(gd_critic, cp)
                    predicted_q = critic_model(obs_aug, act)
                    loss = jnp.mean((predicted_q - target[None, :]) ** 2)
                    return loss, predicted_q

                (critic_loss, predicted_q), critic_grads = (
                    jax.value_and_grad(critic_loss_fn, has_aux=True)(c_p))
                critic_grad_norm = optax.global_norm(critic_grads)
                critic_grads = clip_grads(critic_grads)
                critic_updates, new_c_os = c_opt.update(
                    critic_grads, c_os, c_p)
                new_c_p = optax.apply_updates(c_p, critic_updates)

                def policy_loss_fn(pp):
                    policy_model = nnx.merge(gd_policy, pp)
                    critic_model = nnx.merge(gd_critic, new_c_p)
                    new_action, log_prob = policy_model.sample(
                        obs, action_key, actor_context)
                    q_values = critic_model(obs_aug, new_action)
                    actor_q = q_values.min(axis=0)
                    return (jnp.exp(la) * log_prob - actor_q).mean(), log_prob

                (policy_loss, log_prob), policy_grads = jax.value_and_grad(
                    policy_loss_fn, has_aux=True)(p_p)
                policy_grad_norm = optax.global_norm(policy_grads)
                policy_grads = clip_grads(policy_grads)
                policy_updates, new_p_os = p_opt.update(
                    policy_grads, p_os, p_p)
                new_p_p = optax.apply_updates(p_p, policy_updates)

                alpha_grad = (
                    -jnp.exp(la) * (log_prob.mean() + target_entropy))
                alpha_updates, candidate_a_os = a_opt.update(
                    alpha_grad, a_os, la)
                new_la = jnp.where(auto_alpha, la + alpha_updates, la)
                if alpha_max > 0.0:
                    new_la = jnp.minimum(new_la, jnp.log(alpha_max))
                new_a_os = jax.tree.map(
                    lambda new, old: jnp.where(auto_alpha, new, old),
                    candidate_a_os, a_os)

                new_t_p = jax.tree.map(
                    lambda target_p, critic_p: (
                        target_p * (1.0 - tau) + critic_p * tau),
                    t_p, new_c_p)

                scalar_finite = jnp.all(jnp.isfinite(jnp.stack([
                    critic_loss, policy_loss, context_loss, new_la,
                    critic_grad_norm, policy_grad_norm,
                    context_grad_norm,
                ])))
                parameter_finite = jnp.logical_and(
                    tree_all_finite(new_c_p),
                    jnp.logical_and(
                        tree_all_finite(new_p_p),
                        jnp.logical_and(
                            tree_all_finite(new_x_p),
                            jnp.all(jnp.isfinite(new_proto)))))
                step_finite = jnp.logical_and(
                    scalar_finite, parameter_finite)

                new_carry = (
                    new_c_p, new_t_p, new_p_p, new_x_p, new_la,
                    new_c_os, new_p_os, new_x_os, new_a_os,
                    new_proto, new_proto_valid, key)
                metrics = (
                    critic_loss, policy_loss, context_loss,
                    jnp.exp(new_la), predicted_q.mean(),
                    predicted_q.std(axis=0).mean(), log_prob.mean(),
                    critic_grad_norm, policy_grad_norm,
                    context_grad_norm, step_finite)
                return new_carry, metrics

            initial = (
                critic_params, target_params, policy_params, ctx_params,
                log_alpha, c_opt_state, p_opt_state, x_opt_state,
                a_opt_state, prototypes, prototype_valid, rng_key)
            batches = (
                all_obs, all_prev_act, all_reset_before, all_act, all_rew,
                all_next_obs, all_next_prev_act, all_next_reset_before,
                all_done, all_task_ids)
            return jax.lax.scan(body_fn, initial, batches)

        self._scan_update = _scan_update

    @property
    def alpha(self):
        return jnp.exp(self.log_alpha)

    def reset_recurrent_context(self):
        if not getattr(self, "uses_recurrent_context", False):
            return
        self._online_hidden = self.context_net.initial_hidden((1,))
        self._online_previous_action = jnp.zeros(
            (1, self.act_dim), dtype=jnp.float32)

    def snapshot_recurrent_context(self):
        if not self.uses_recurrent_context:
            return None
        return (
            jnp.array(self._online_hidden),
            jnp.array(self._online_previous_action),
        )

    def restore_recurrent_context(self, snapshot):
        if not self.uses_recurrent_context or snapshot is None:
            return
        self._online_hidden, self._online_previous_action = snapshot

    def finish_recurrent_step(self, done):
        if self.uses_recurrent_context and bool(done):
            self.reset_recurrent_context()

    def select_action(self, obs, deterministic=False):
        obs_jax = (
            jnp.array(obs)[None]
            if np.asarray(obs).ndim == 1 else jnp.array(obs))
        if self.uses_recurrent_context:
            self._online_hidden, ep = self.context_net.step(
                self._online_hidden, obs_jax,
                self._online_previous_action)
            if (not deterministic
                    and float(getattr(
                        self.config, "escp_bottleneck_sigma", 0.0)) > 0.0):
                ep = ep + float(self.config.escp_bottleneck_sigma) * (
                    jax.random.normal(self.rngs.params(), ep.shape))
        else:
            ep = self.context_net(obs_jax)
        if deterministic:
            action = self.policy.deterministic(obs_jax, ep)[0]
        else:
            action, _ = self.policy.sample(
                obs_jax, self.rngs.params(), ep)
            action = action[0]
        if self.uses_recurrent_context:
            self._online_previous_action = action[None]
        return np.asarray(action)

    def _context_phase(self, current_iter: int) -> tuple[bool, bool, int]:
        collected_steps = (
            int(getattr(self.config, "initial_random_steps", 0))
            + (int(current_iter) + 1) * int(self.config.samples_per_iter))
        changing_period = max(int(self.config.changing_period), 1)
        observed_tasks = min(
            int(self.config.task_num),
            1 + collected_steps // changing_period)
        context_train = bool(
            collected_steps >= int(getattr(
                self.config, "escp_context_min_steps", 0))
            and observed_tasks >= int(getattr(
                self.config, "escp_context_min_tasks", 0)))
        if self.uses_recurrent_context:
            # Released ESCP conditions the policy on its causal probe before
            # the delayed RMDM optimization phase begins.
            context_active = bool(
                current_iter >= self.config.context_warmup_iters)
        else:
            context_active = bool(
                current_iter >= self.config.context_warmup_iters
                and context_train)
        return context_active, context_train, observed_tasks

    def multi_update(self, stacked_batch, current_iter=0, **kwargs):
        del kwargs
        rng_key = self.rngs.params()
        context_active, context_train, observed_tasks = self._context_phase(
            current_iter)

        critic_params = nnx.state(self.critic, nnx.Param)
        target_params = nnx.state(self.target_critic, nnx.Param)
        policy_params = nnx.state(self.policy, nnx.Param)
        context_params = nnx.state(self.context_net, nnx.Param)

        obs = stacked_batch["obs"]
        act = stacked_batch["act"]
        rew = stacked_batch["rew"]
        next_obs = stacked_batch["next_obs"]
        done = stacked_batch["done"]
        task_ids = stacked_batch["task_id"]

        if self.uses_recurrent_context:
            final, metrics = self._scan_update(
                critic_params, target_params, policy_params, context_params,
                self.log_alpha, self.critic_opt_state,
                self.policy_opt_state, self.context_opt_state,
                self.alpha_opt_state, self.context_prototypes,
                self.context_prototype_valid,
                obs, stacked_batch["prev_act"],
                stacked_batch["reset_before"], act, rew, next_obs,
                stacked_batch["next_prev_act"],
                stacked_batch["next_reset_before"], done, task_ids,
                rng_key, jnp.asarray(context_active),
                jnp.asarray(context_train))
            (new_critic, new_target, new_policy, new_context, new_log_alpha,
             new_critic_opt_state, new_policy_opt_state,
             new_context_opt_state, new_alpha_opt_state,
             new_prototypes, new_prototype_valid, _) = final
        else:
            final, metrics = self._scan_update(
                critic_params, target_params, policy_params, context_params,
                self.log_alpha, self.critic_opt_state,
                self.policy_opt_state, self.context_opt_state,
                self.alpha_opt_state, obs, act, rew, next_obs, done,
                task_ids, rng_key, jnp.asarray(context_active),
                jnp.asarray(context_train))
            (new_critic, new_target, new_policy, new_context, new_log_alpha,
             new_critic_opt_state, new_policy_opt_state,
             new_context_opt_state, new_alpha_opt_state, _) = final
        (critic_loss, policy_loss, context_loss, alpha, q_mean, q_std,
         log_prob, critic_grad_norm, policy_grad_norm, context_grad_norm,
         finite) = metrics

        finite_np = np.asarray(finite, dtype=bool)
        if (bool(getattr(self.config, "escp_finite_guard", True))
                and not bool(np.all(finite_np))):
            first_bad = int(np.flatnonzero(~finite_np)[0])

            def metric_at(values):
                return float(np.asarray(values)[first_bad])

            details = {
                "critic_loss": metric_at(critic_loss),
                "policy_loss": metric_at(policy_loss),
                "rmdm_loss": metric_at(context_loss),
                "alpha": metric_at(alpha),
                "critic_grad_norm": metric_at(critic_grad_norm),
                "policy_grad_norm": metric_at(policy_grad_norm),
                "context_grad_norm": metric_at(context_grad_norm),
            }
            raise FloatingPointError(
                "ESCP non-finite update at "
                f"global_update={self.update_count + first_bad}, "
                f"scan_index={first_bad}, iter={current_iter}, "
                f"target_mode={self.config.escp_target_mode}, "
                f"actor_mode={self.config.escp_actor_mode}, "
                f"details={details}")

        self.critic_opt_state = new_critic_opt_state
        self.policy_opt_state = new_policy_opt_state
        self.context_opt_state = new_context_opt_state
        self.alpha_opt_state = new_alpha_opt_state
        nnx.update(self.critic, new_critic)
        nnx.update(self.target_critic, new_target)
        nnx.update(self.policy, new_policy)
        nnx.update(self.context_net, new_context)
        self.log_alpha = new_log_alpha
        if self.uses_recurrent_context:
            self.context_prototypes = new_prototypes
            self.context_prototype_valid = new_prototype_valid
        self.update_count += int(obs.shape[0])

        return {
            "critic_loss": float(critic_loss.mean()),
            "policy_loss": float(policy_loss.mean()),
            "rmdm_loss": float(context_loss.mean()),
            "alpha": float(alpha[-1]),
            "q_mean": float(q_mean.mean()),
            "q_std_mean": float(q_std.mean()),
            "log_prob": float(log_prob.mean()),
            "critic_grad_norm": float(critic_grad_norm.mean()),
            "policy_grad_norm": float(policy_grad_norm.mean()),
            "context_grad_norm": float(context_grad_norm.mean()),
            "finite_update_rate": float(finite.mean()),
            "context_train": context_train,
            "context_active": context_active,
            "observed_task_count_estimate": observed_tasks,
        }

    def context_checkpoint_signature(self):
        return {
            "schema": "escp.context.v2",
            "mode": self.context_mode,
            "obs_dim": int(self.obs_dim),
            "act_dim": int(self.act_dim),
            "ep_dim": int(self.config.ep_dim),
            "history_length": int(getattr(
                self.config, "escp_history_length", 16)),
        }

    def checkpoint_state(self):
        if not self.uses_recurrent_context:
            return {"schema": "escp.agent.v1", "context_mode": "state_mlp"}
        return {
            "schema": "escp.agent.v2",
            "context_mode": "recurrent",
            "context_prototypes": np.asarray(self.context_prototypes),
            "context_prototype_valid": np.asarray(
                self.context_prototype_valid),
        }

    def load_checkpoint_state(self, state):
        if not self.uses_recurrent_context:
            return
        if state.get("context_mode") != "recurrent":
            raise ValueError(
                "recurrent ESCP cannot resume a non-recurrent checkpoint")
        prototypes = jnp.asarray(
            state["context_prototypes"], dtype=jnp.float32)
        valid = jnp.asarray(
            state["context_prototype_valid"], dtype=jnp.bool_)
        if prototypes.shape != self.context_prototypes.shape:
            raise ValueError(
                "ESCP context prototype shape changed across resume")
        if valid.shape != self.context_prototype_valid.shape:
            raise ValueError(
                "ESCP context prototype-valid shape changed across resume")
        self.context_prototypes = prototypes
        self.context_prototype_valid = valid
        self.reset_recurrent_context()
