import gtimer as gt
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from torch.cuda.amp import autocast, GradScaler

class SACTrainer(TorchTrainer):

    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            discount=0.99,
            reward_scale=1.0,
            alpha=1.,
            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,
            soft_target_tau=1e-2,
            target_update_period=1,
            clip_norm=0.,
            use_automatic_entropy_tuning=True,
            target_entropy=None,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )
        else:
            self.alpha = alpha

        self.qf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.clip_norm = clip_norm
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.scaler = GradScaler()

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        gt.stamp('preback_start', unique=False)
        """
        Update Alpha
        """
        with autocast():
            new_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
                obs,
                reparameterize=True,
                return_log_prob=True,
            )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            self.scaler.scale(alpha_loss).backward()
            self.scaler.step(self.alpha_optimizer)
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self.alpha
        """
        Update QF
        """
        with autocast():
            with torch.no_grad():
                new_next_actions, _, _, new_log_pi, *_ = self.policy(
                    next_obs,
                    reparameterize=True,
                    return_log_prob=True,
                )
                target_q_values = torch.min(
                    self.target_qf1(next_obs, new_next_actions),
                    self.target_qf2(next_obs, new_next_actions),
                ) - alpha * new_log_pi

                q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
            q1_pred = self.qf1(obs, actions)
            q2_pred = self.qf2(obs, actions)
            qf1_loss = self.qf_criterion(q1_pred, q_target)
            qf2_loss = self.qf_criterion(q2_pred, q_target)
        gt.stamp('preback_qf', unique=False)

        self.qf1_optimizer.zero_grad()
        self.scaler.scale(qf1_loss).backward()
        self.scaler.step(self.qf1_optimizer)
        gt.stamp('backward_qf1', unique=False)

        self.qf2_optimizer.zero_grad()
        self.scaler.scale(qf2_loss).backward()
        self.scaler.step(self.qf2_optimizer)
        gt.stamp('backward_qf2', unique=False)
        """
        Update Policy
        """
        with autocast():
            q_new_actions = torch.min(
                self.qf1(obs, new_actions),
                self.qf2(obs, new_actions),
            )
            policy_loss = (alpha * log_pi.float() - q_new_actions).mean()
        gt.stamp('preback_policy', unique=False)

        self.policy_optimizer.zero_grad()
        self.scaler.scale(policy_loss).backward()
        if self.clip_norm > 0:
            self.scaler.unscale_(self.policy_optimizer)
        policy_grad = ptu.fast_clip_grad_norm(self.policy.parameters(), self.clip_norm)
        self.scaler.step(self.policy_optimizer)
        
        self.scaler.update()
        gt.stamp('backward_policy', unique=False)
        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False

            self.eval_statistics['QF1 Loss'] = qf1_loss.item()
            self.eval_statistics['QF2 Loss'] = qf2_loss.item()
            self.eval_statistics['Policy Loss'] = policy_loss.item()
            self.eval_statistics['Policy Grad'] = policy_grad
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred.float()),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred.float()),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target.float()),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi.float()),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean.float()),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std.float()),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy.state_dict(),
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            target_qf1=self.qf1.state_dict(),
            target_qf2=self.qf2.state_dict(),
        )
