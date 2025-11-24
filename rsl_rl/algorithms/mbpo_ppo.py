# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
import warnings

from rsl_rl.modules import ActorCritic
from rsl_rl.modules import SystemDynamicsEnsemble
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable
from rsl_rl.storage.replay_buffer import ReplayBuffer

from rsl_rl.algorithms.ppo import PPO


class MBPOPPO(PPO):
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    actor_critic: ActorCritic
    """The actor critic module."""
    system_dynamics: SystemDynamicsEnsemble
    """The system dynamics ensemble module."""
    
    def __init__(
        self,
        policy,
        system_dynamics,
        state_normalizer,
        action_normalizer,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        policy_learning_rate=0.001,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # System dynamics parameters
        system_dynamics_learning_rate=1e-3,
        system_dynamics_weight_decay=0.0,
        system_dynamics_forecast_horizon=1,
        system_dynamics_loss_weights={"state": 1.0, "sequence": 1.0, "bound": 1.0, "extension": 1.0, "contact": 1.0, "termination": 1.0},
        system_dynamics_num_mini_batches=10,
        system_dynamics_mini_batch_size=1000,
        system_dynamics_replay_buffer_size=10000,
        system_dynamics_num_eval_trajectories=10,
        system_dynamics_len_eval_trajectory=400,
        system_dynamics_eval_traj_noise_scale=[0.1, 0.2, 0.4, 0.5, 0.8],
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        super().__init__(
            policy=policy,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=policy_learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )
        
        self.system_dynamics_learning_rate = system_dynamics_learning_rate
        self.system_dynamics_weight_decay = system_dynamics_weight_decay
        self.imagination_storage = None  # initialized later
        # System dynamics components
        self.system_dynamics = system_dynamics
        self.system_dynamics.to(self.device)
        self.system_replay_buffer = ReplayBuffer(
            [system_dynamics.state_dim, system_dynamics.action_dim, system_dynamics.extension_dim, system_dynamics.contact_dim, system_dynamics.termination_dim],
            system_dynamics_replay_buffer_size,
            device
            )
        self.system_dynamics_optimizer = optim.Adam(self.system_dynamics.parameters(), lr=system_dynamics_learning_rate, weight_decay=system_dynamics_weight_decay)
        
        # System dynamics parameters
        self.system_dynamics_forecast_horizon = system_dynamics_forecast_horizon
        self.system_dynamics_loss_weights = system_dynamics_loss_weights
        self.system_dynamics_num_mini_batches = system_dynamics_num_mini_batches
        self.system_dynamics_mini_batch_size = system_dynamics_mini_batch_size
        self.system_dynamics_num_eval_trajectories = system_dynamics_num_eval_trajectories
        self.system_dynamics_len_eval_trajectory = system_dynamics_len_eval_trajectory
        self.system_dynamics_eval_traj_noise_scale = system_dynamics_eval_traj_noise_scale

        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape, imagination=False):
        if imagination:
            imagination_obs = TensorDict(
                {
                    key: torch.zeros(num_envs, value.shape[1], device=value.device) for key, value in obs.items()
                },
                batch_size=[num_envs],
                device=obs.device,
            )
            self.imagination_storage = RolloutStorage(
                training_type,
                num_envs,
                num_transitions_per_env,
                imagination_obs,
                actions_shape,
                self.device,
            )
        else:
            super().init_storage(training_type, num_envs, num_transitions_per_env, obs, actions_shape)
    
    def process_env_step(self, obs, rewards, dones, extras, imagination=False):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # record the transition
        if imagination:
            self.imagination_storage.add_transitions(self.transition)
        else:
            self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def fill_history_buffer(self, obs):
        system_state = obs["system_state"]
        system_action = obs["system_action"]
        system_extension = obs.get("system_extension")
        system_contact = obs.get("system_contact")
        system_termination = obs.get("system_termination")
        system_state = self.state_normalizer(system_state)
        system_action = self.action_normalizer(system_action)

        self.system_replay_buffer.insert(
            [
                system_state.unsqueeze(1),
                system_action.unsqueeze(1),
                system_extension.unsqueeze(1) if system_extension is not None else None,
                system_contact.unsqueeze(1) if system_contact is not None else None,
                system_termination.unsqueeze(1) if system_termination is not None else None,
                ]
            )

    def compute_returns(self, obs, imagination=False):
        if imagination:
            # compute value for the last step
            last_values = self.policy.evaluate(obs).detach()
            self.imagination_storage.compute_returns(
                last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
            )
        else:
            super().compute_returns(obs)

    def update_system_dynamics(self):
        mean_system_state_loss = 0
        mean_system_sequence_loss = 0
        mean_system_bound_loss = 0
        mean_system_kl_loss = 0
        mean_system_extension_loss = 0
        mean_system_contact_loss = 0
        mean_system_termination_loss = 0
        system_generator = self.system_replay_buffer.mini_batch_generator(
            self.system_dynamics.history_horizon + self.system_dynamics_forecast_horizon,
            self.system_dynamics_num_mini_batches,
            self.system_dynamics_mini_batch_size,
        )
        for system_state_batch, system_action_batch, system_extension_batch, system_contact_batch, system_termination_batch in system_generator:
            self.system_dynamics.reset()
            state_loss, sequence_loss, bound_loss, kl_loss, extension_loss, contact_loss, termination_loss = self.system_dynamics.compute_loss(
                system_state_batch,
                system_action_batch,
                system_extension_batch,
                system_contact_batch,
                system_termination_batch,
                bootstrap=True
            )
            loss = (
                self.system_dynamics_loss_weights["state"] * state_loss
                + self.system_dynamics_loss_weights["sequence"] * sequence_loss
                + self.system_dynamics_loss_weights["bound"] * bound_loss
                + self.system_dynamics_loss_weights["kl"] * kl_loss
                + self.system_dynamics_loss_weights["extension"] * extension_loss
                + self.system_dynamics_loss_weights["contact"] * contact_loss
                + self.system_dynamics_loss_weights["termination"] * termination_loss
            )
            self.system_dynamics_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.system_dynamics.parameters(), self.max_grad_norm)
            self.system_dynamics_optimizer.step()
            mean_system_state_loss += state_loss.item()
            mean_system_sequence_loss += sequence_loss.item()
            mean_system_bound_loss += bound_loss.item()
            mean_system_kl_loss += kl_loss.item()
            mean_system_extension_loss += extension_loss.item()
            mean_system_contact_loss += contact_loss.item()
            mean_system_termination_loss += termination_loss.item()
    
        system_dynamics_num_updates = self.system_dynamics_num_mini_batches
        mean_system_state_loss /= system_dynamics_num_updates
        mean_system_sequence_loss /= system_dynamics_num_updates
        mean_system_bound_loss /= system_dynamics_num_updates
        mean_system_kl_loss /= system_dynamics_num_updates
        mean_system_extension_loss /= system_dynamics_num_updates
        mean_system_contact_loss /= system_dynamics_num_updates
        mean_system_termination_loss /= system_dynamics_num_updates
        return mean_system_state_loss, mean_system_sequence_loss, mean_system_bound_loss, mean_system_kl_loss, mean_system_extension_loss, mean_system_contact_loss, mean_system_termination_loss
    
    def evaluate_system_dynamics(self):
        system_generator = self.system_replay_buffer.mini_batch_generator(
            self.system_dynamics_len_eval_trajectory,
            1,
            self.system_dynamics_num_eval_trajectories,
        )
        state_traj, action_traj, extension_traj, contact_traj, termination_traj = next(system_generator)
        state_traj_pred, _, _, action_traj_pred, extension_traj_pred, contact_traj_pred, termination_traj_pred = self.system_dynamics_autoregressive_prediction(state_traj, action_traj, extension_traj, contact_traj, termination_traj)
        traj_autoregressive_error = ((state_traj_pred[:, self.system_dynamics.history_horizon:] - state_traj[:, self.system_dynamics.history_horizon:]).abs().sum(dim=-1) / state_traj[:, self.system_dynamics.history_horizon:].abs().sum(dim=-1)).mean().item()
        traj_autoregressive_error_noised_dict = {}
        for noise_scale in self.system_dynamics_eval_traj_noise_scale:
            state_traj_noised = state_traj + torch.randn_like(state_traj) * noise_scale
            action_traj_noised = action_traj + torch.randn_like(action_traj) * noise_scale
            state_traj_pred_noised, _, _, _, _, _, _ = self.system_dynamics_autoregressive_prediction(state_traj_noised, action_traj_noised, extension_traj, contact_traj, termination_traj)
            traj_autoregressive_error_noised = ((state_traj_pred_noised[:, self.system_dynamics.history_horizon:] - state_traj_noised[:, self.system_dynamics.history_horizon:]).abs().sum(dim=-1) / state_traj_noised[:, self.system_dynamics.history_horizon:].abs().sum(dim=-1)).mean().item()
            traj_autoregressive_error_noised_dict[noise_scale] = traj_autoregressive_error_noised

        return state_traj, action_traj, extension_traj, contact_traj, termination_traj, state_traj_pred, action_traj_pred, extension_traj_pred, contact_traj_pred, termination_traj_pred, traj_autoregressive_error, traj_autoregressive_error_noised_dict

    def system_dynamics_autoregressive_prediction(self, state_traj, action_traj, extension_traj, contact_traj, termination_traj):
        state_traj_pred = torch.zeros_like(state_traj, device=self.device)
        aleatoric_uncertainty_traj_pred = torch.zeros(state_traj.shape[0], state_traj.shape[1], device=self.device)
        epistemic_uncertainty_traj_pred = torch.zeros(state_traj.shape[0], state_traj.shape[1], device=self.device)
        action_traj_pred = action_traj.clone()
        extension_traj_pred = torch.zeros_like(extension_traj, device=self.device) if extension_traj is not None else None
        contact_traj_pred = torch.zeros_like(contact_traj, device=self.device) if contact_traj is not None else None
        termination_traj_pred = torch.zeros_like(termination_traj, device=self.device) if termination_traj is not None else None
        
        state_traj_pred[:, :self.system_dynamics.history_horizon] = state_traj[:, :self.system_dynamics.history_horizon]
        if extension_traj_pred is not None:
            extension_traj_pred[:, :self.system_dynamics.history_horizon] = extension_traj[:, :self.system_dynamics.history_horizon]
        if contact_traj_pred is not None:
            contact_traj_pred[:, :self.system_dynamics.history_horizon] = contact_traj[:, :self.system_dynamics.history_horizon]
        if termination_traj_pred is not None:
            termination_traj_pred[:, :self.system_dynamics.history_horizon] = termination_traj[:, :self.system_dynamics.history_horizon]

        self.system_dynamics.reset()
        with torch.inference_mode():
            for i in range(self.system_dynamics.history_horizon, self.system_dynamics_len_eval_trajectory):
                if self.system_dynamics.architecture_config["type"] in ["rnn", "rssm"] and i > self.system_dynamics.history_horizon:
                    state_input = state_traj_pred[:, i - 1:i]
                    action_input = action_traj_pred[:, i - 1:i]
                else:
                    state_input = state_traj_pred[:, i - self.system_dynamics.history_horizon:i]
                    action_input = action_traj_pred[:, i - self.system_dynamics.history_horizon:i]
                state_pred, aleatoric_uncertainty, epistemic_uncertainty, extension_pred, contact_pred, termination_pred = self.system_dynamics.forward(state_input, action_input)
                state_traj_pred[:, i] = state_pred
                aleatoric_uncertainty_traj_pred[:, i] = aleatoric_uncertainty
                epistemic_uncertainty_traj_pred[:, i] = epistemic_uncertainty
                if extension_traj_pred is not None and extension_pred is not None:
                    extension_traj_pred[:, i] = extension_pred
                if contact_traj_pred is not None and contact_pred is not None:
                    contact_traj_pred[:, i] = torch.sigmoid(contact_pred).round().int()
                if termination_traj_pred is not None and termination_pred is not None:
                    termination_traj_pred[:, i] = torch.sigmoid(termination_pred).round().int()
        return state_traj_pred, aleatoric_uncertainty_traj_pred, epistemic_uncertainty_traj_pred, action_traj_pred, extension_traj_pred, contact_traj_pred, termination_traj_pred


    def prepare_imagination(self):
        imagination_generator = self.system_replay_buffer.mini_batch_generator(self.system_dynamics.history_horizon, 1, self.imagination_storage.num_envs)
        imagination_state_history, imagination_action_history = next(imagination_generator)[:2]
        return imagination_state_history, imagination_action_history

    def mini_batch_generator_combined(self, real_storage, imagination_storage):
        real_generator = real_storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        imagination_generator = imagination_storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for real_batch, imagination_batch in zip(real_generator, imagination_generator):
            combined_batch = []
            for real_batch_term, imagination_batch_term in zip(real_batch, imagination_batch):
                if isinstance(real_batch_term, torch.Tensor) and isinstance(imagination_batch_term, torch.Tensor):
                    combined_term = torch.cat([real_batch_term, imagination_batch_term], dim=0)
                elif isinstance(real_batch_term, TensorDict) and isinstance(imagination_batch_term, TensorDict):
                    combined_term = TensorDict(
                        {
                            key: torch.cat([real_batch_term[key], imagination_batch_term[key]], dim=0)
                            for key in real_batch_term.keys()
                        },
                        batch_size=[real_batch_term.batch_size[0] + imagination_batch_term.batch_size[0]],
                        device=real_batch_term.device,
                    )
                else:
                    combined_term = real_batch_term
                combined_batch.append(combined_term)
            yield tuple(combined_batch)

    def update(self, imagination=False):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches
        if imagination:
            generator = self.mini_batch_generator_combined(self.storage, self.imagination_storage)
        else:
            if self.policy.is_recurrent:
                generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            else:
                generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            # we assume policy group is always there and needs augmentation
            original_batch_size = obs_batch["policy"].shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(  # TODO: needs changes on the isaac lab side
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                # compute number of augmentations per sample
                # we assume policy group is always there and needs augmentation
                num_aug = int(obs_batch["policy"].shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            # TODO: Move this processing to inside RND module.
            if self.rnd:
                # extract the rnd_state
                # TODO: Check if we still need torch no grad. It is just an affine transformation.
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()
        if imagination:
            self.imagination_storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict
