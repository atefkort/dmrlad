# This code is based on rlkit sac_v2 implementation.

from collections import OrderedDict
import os

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import gtimer as gt


import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict

import matplotlib.pyplot as plt
from .risk import distortion_de
from .utils import LinearSchedule

def quantile_regression_loss(input, target, tau, weight):
    """
    input: (N, T)
    target: (N, T)
    tau: (N, T)
    """
    input = input.unsqueeze(-1)
    target = target.detach().unsqueeze(-2)
    tau = tau.detach().unsqueeze(-1)
    weight = weight.detach().unsqueeze(-2)
    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    L = nn.functional.smooth_l1_loss(expanded_input, expanded_target, reduction="none")  # (N, T, T)
    sign = torch.sign(expanded_input - expanded_target) / 2. + 0.5
    rho = torch.abs(tau - sign) * L * weight
    return rho.sum(dim=-1).mean()

class PolicyTrainer:
    def __init__(
            self,
            policy,
            target_policy,
            zf1,
            zf2,
            target_zf1,
            target_zf2,
            alpha_net,
            
            replay_buffer,
            batch_size,

            env_action_space,
            data_usage_sac,
            
            fp=None,
            target_fp=None,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=3e-4,
            zf_lr=3e-4,
            tau_type='iqn',
            fp_lr=1e-5,
            num_quantiles=32,
            risk_type='neutral',
            risk_param=0.,
            risk_param_final=None,
            risk_schedule_timesteps=1,
            clip_norm=0.,
            optimizer_class=optim.Adam,

            soft_target_tau=5e-3,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            use_parametrized_alpha=False,
            target_entropy=None,
            target_entropy_factor=1.0,
            alpha=1.0

    ):
        super().__init__()
        self.policy = policy
        self.target_policy = target_policy
        self.zf1 = zf1
        self.zf2 = zf2
        self.target_zf1 = target_zf1
        self.target_zf2 = target_zf2
        self.alpha_net = alpha_net
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.tau_type = tau_type
        self.num_quantiles = num_quantiles

        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

        self.env_action_space = env_action_space
        self.data_usage_sac= data_usage_sac

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.use_parametrized_alpha = use_parametrized_alpha
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -self.env_action_space  # heuristic value from Tuomas
            self.target_entropy = self.target_entropy * target_entropy_factor

            if self.use_parametrized_alpha:
                self.alpha_optimizer = optimizer_class(
                    self.alpha_net.parameters(),
                    lr=policy_lr,
                )
            else:
                self.log_alpha = ptu.zeros(1, requires_grad=True)
                self.alpha_optimizer = optimizer_class(
                    [self.log_alpha],
                    lr=policy_lr,
                )
        self._alpha = alpha

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.zf_criterion = quantile_regression_loss

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.zf1_optimizer = optimizer_class(
            self.zf1.parameters(),
            lr=zf_lr,
        )
        self.zf2_optimizer = optimizer_class(
            self.zf2.parameters(),
            lr=zf_lr,
        )

        self.fp = fp
        self.target_fp = target_fp
        if self.tau_type == 'fqf':
            self.fp_optimizer = optimizer_class(
                self.fp.parameters(),
                lr=fp_lr,
            )

        self.risk_type = risk_type
        self.risk_schedule = LinearSchedule(risk_schedule_timesteps, risk_param,
                                            risk_param if risk_param_final is None else risk_param_final)

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.clip_norm = clip_norm

    def get_tau(self, obs, actions, fp=None):
        if self.tau_type == 'fix':
            presum_tau = ptu.zeros(len(actions), self.num_quantiles) + 1. / self.num_quantiles
        elif self.tau_type == 'iqn':  # add 0.1 to prevent tau getting too close
            presum_tau = ptu.rand(len(actions), self.num_quantiles) + 0.1
            presum_tau /= presum_tau.sum(dim=-1, keepdims=True)
        elif self.tau_type == 'fqf':
            if fp is None:
                fp = self.fp
            presum_tau = fp(obs, actions)
        tau = torch.cumsum(presum_tau, dim=1)  # (N, T), note that they are tau1...tauN in the paper
        with torch.no_grad():
            tau_hat = ptu.zeros_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
        return tau, tau_hat, presum_tau

    def train(self, epochs):
        gt.stamp('pt_train_start')
        indices = np.array(self.replay_buffer.get_allowed_points())
        if self.data_usage_sac == 'tree_sampling':
            indices = np.random.permutation(indices)
        policy_losses = []
        alphas = []
        log_pis = []
        for epoch in range(epochs):
            policy_loss, alpha, log_pi = self.training_step(indices, epoch)
            policy_losses.append(policy_loss/1.0)
            alphas.append(alpha / 1.0)
            log_pis.append((-1) * log_pi.mean() / 1.0)
            if epoch % 100 == 0 and int(os.environ['DEBUG']) == 1:
                print("Epoch: " + str(epoch) + ", policy loss: " + str(policy_losses[-1]))

        if int(os.environ['PLOT']) == 1:
            plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(list(range(len(policy_losses))), np.array(policy_losses), label="Policy loss")
            plt.xlim(left=0)
            plt.legend()
            #plt.ylim(bottom=0)
            plt.subplot(3, 1, 2)
            plt.plot(list(range(len(alphas))), np.array(alphas), label="alphas")
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.plot(list(range(len(log_pis))), np.array(log_pis), label="Entropy")
            plt.legend()
            plt.show(block=False)

        self.eval_statistics['policy_train_steps_total'] = self._n_train_steps_total
        self.end_epoch(epoch)

        return policy_losses[-1], self.get_diagnostics()

    def training_step(self, indices, step):
        # get data from replay buffer
        if step == 0:
            gt.stamp('pt_before_sample')
        batch = self.replay_buffer.sample_sac_data_batch(indices, self.batch_size)
        if step == 0:
            gt.stamp('pt_sample')

        rewards = ptu.from_numpy(batch['rewards'])
        terminals = ptu.from_numpy(batch['terminals'])
        obs = ptu.from_numpy(batch['observations'])
        actions = ptu.from_numpy(batch['actions'])
        next_obs = ptu.from_numpy(batch['next_observations'])
        task_z = ptu.from_numpy(batch['task_indicators'])
        new_task_z = ptu.from_numpy(batch['next_task_indicators'])
        if step == 0:
            gt.stamp('pt_to_torch')

        #for debug
        #task_z = torch.zeros_like(task_z)
        #new_task_z = torch.zeros_like(new_task_z)
        #task_z = torch.from_numpy(batch['true_tasks'])
        #new_task_z = torch.cat([task_z[1:,:], task_z[-1,:].view(1,1)])
        new_task_z = task_z.clone().detach()

        # Variant 1: train the SAC as if there was no encoder and the state is just extended to be [state , z]
        obs = torch.cat((obs, task_z), dim=1)
        next_obs = torch.cat((next_obs, new_task_z), dim=1)

        """
        Policy and Alpha Loss
        """
        new_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            if self.use_parametrized_alpha:
                self.log_alpha = self.alpha_net(task_z)
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            if self.use_parametrized_alpha:
                alpha = self.log_alpha.exp().detach()
            else:
                alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self._alpha

        if step == 0:
            gt.stamp('pt_alpha')

        """
        Update ZF
        """
        with torch.no_grad():
            new_next_actions, _, _, new_log_pi, *_ = self.target_policy(
                next_obs,
                reparameterize=True,
                return_log_prob=True,
            )
            next_tau, next_tau_hat, next_presum_tau = self.get_tau(next_obs, new_next_actions, fp=self.target_fp)
            target_z1_values = self.target_zf1(next_obs, new_next_actions, next_tau_hat)
            target_z2_values = self.target_zf2(next_obs, new_next_actions, next_tau_hat)
            target_z_values = torch.min(target_z1_values, target_z2_values) - alpha * new_log_pi
            z_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_z_values

        tau, tau_hat, presum_tau = self.get_tau(obs, actions, fp=self.fp)
        z1_pred = self.zf1(obs, actions, tau_hat)
        z2_pred = self.zf2(obs, actions, tau_hat)
        zf1_loss = self.zf_criterion(z1_pred, z_target, tau_hat, next_presum_tau)
        zf2_loss = self.zf_criterion(z2_pred, z_target, tau_hat, next_presum_tau)
        gt.stamp('preback_zf', unique=False)

        self.zf1_optimizer.zero_grad()
        zf1_loss.backward()
        self.zf1_optimizer.step()
        gt.stamp('backward_zf1', unique=False)

        self.zf2_optimizer.zero_grad()
        zf2_loss.backward()
        self.zf2_optimizer.step()
        gt.stamp('backward_zf2', unique=False)

        """
        Update FP
        """
        if self.tau_type == 'fqf':
            with torch.no_grad():
                dWdtau = 0.5 * (2 * self.zf1(obs, actions, tau[:, :-1]) - z1_pred[:, :-1] - z1_pred[:, 1:] +
                                2 * self.zf2(obs, actions, tau[:, :-1]) - z2_pred[:, :-1] - z2_pred[:, 1:])
                dWdtau /= dWdtau.shape[0]  # (N, T-1)
            gt.stamp('preback_fp', unique=False)

            self.fp_optimizer.zero_grad()
            tau[:, :-1].backward(gradient=dWdtau)
            self.fp_optimizer.step()
            gt.stamp('backward_fp', unique=False)

        """
        Update Policy
        """
        risk_param = self.risk_schedule(self._n_train_steps_total)

        if self.risk_type == 'VaR':
            tau_ = ptu.ones_like(rewards) * risk_param
            q1_new_actions = self.zf1(obs, new_actions, tau_)
            q2_new_actions = self.zf2(obs, new_actions, tau_)
        else:
            with torch.no_grad():
                new_tau, new_tau_hat, new_presum_tau = self.get_tau(obs, new_actions, fp=self.fp)
            z1_new_actions = self.zf1(obs, new_actions, new_tau_hat)
            z2_new_actions = self.zf2(obs, new_actions, new_tau_hat)
            if self.risk_type in ['neutral', 'std']:
                q1_new_actions = torch.sum(new_presum_tau * z1_new_actions, dim=1, keepdims=True)
                q2_new_actions = torch.sum(new_presum_tau * z2_new_actions, dim=1, keepdims=True)
                if self.risk_type == 'std':
                    q1_std = new_presum_tau * (z1_new_actions - q1_new_actions).pow(2)
                    q2_std = new_presum_tau * (z2_new_actions - q2_new_actions).pow(2)
                    q1_new_actions -= risk_param * q1_std.sum(dim=1, keepdims=True).sqrt()
                    q2_new_actions -= risk_param * q2_std.sum(dim=1, keepdims=True).sqrt()
            else:
                with torch.no_grad():
                    risk_weights = distortion_de(new_tau_hat, self.risk_type, risk_param)
                q1_new_actions = torch.sum(risk_weights * new_presum_tau * z1_new_actions, dim=1, keepdims=True)
                q2_new_actions = torch.sum(risk_weights * new_presum_tau * z2_new_actions, dim=1, keepdims=True)
        q_new_actions = torch.min(q1_new_actions, q2_new_actions)

        policy_loss = (alpha * log_pi - q_new_actions).mean()
        gt.stamp('preback_policy', unique=False)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_grad = ptu.fast_clip_grad_norm(self.policy.parameters(), self.clip_norm)
        self.policy_optimizer.step()

        if step == 0:
            gt.stamp('pt_policy_update')

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.policy, self.target_policy, self.soft_target_tau)
            ptu.soft_update_from_to(self.zf1, self.target_zf1, self.soft_target_tau)
            ptu.soft_update_from_to(self.zf2, self.target_zf2, self.soft_target_tau)
            if self.tau_type == 'fqf':
                ptu.soft_update_from_to(self.fp, self.target_fp, self.soft_target_tau)

        if step == 0:
            gt.stamp('pt_q_softupdate')

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['ZF1 Loss'] = zf1_loss.item()
            self.eval_statistics['ZF2 Loss'] = zf2_loss.item()
            self.eval_statistics['Policy Loss'] = policy_loss.item()
            self.eval_statistics['Policy Grad'] = policy_grad
            self.eval_statistics.update(create_stats_ordered_dict(
                'Z1 Predictions',
                ptu.get_numpy(z1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Z2 Predictions',
                ptu.get_numpy(z2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Z Targets',
                ptu.get_numpy(z_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.mean().item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.mean().item()
        self._n_train_steps_total += 1

        if step == 0:
            gt.stamp('pt_statistics')

        return ptu.get_numpy(policy_loss), ptu.get_numpy(alpha), ptu.get_numpy(log_pi)

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.target_policy,
            self.zf1,
            self.zf2,
            self.target_zf1,
            self.target_zf2,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            zf1=self.zf1,
            zf2=self.zf2,
            target_zf1=self.zf1,
            target_zf2=self.zf2,
        )

