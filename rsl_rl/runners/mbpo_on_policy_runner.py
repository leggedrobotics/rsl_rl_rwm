# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
import warnings
from collections import deque

import rsl_rl
from rsl_rl.algorithms import PPO, MBPOPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config, EmpiricalNormalization, SystemDynamicsEnsemble
from rsl_rl.utils import store_code_state
from rsl_rl.modules.plotter import Plotter
import matplotlib.pyplot as plt

from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class MBPOOnPolicyRunner(OnPolicyRunner):
    """On-policy runner for training and evaluation of actor-critic methods."""

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        self._prepare_logging_writer()

        self.plotter = Plotter()
        self.fig0, self.ax0 = plt.subplots(1, 1)
        self.fig1, self.ax1 = plt.subplots(len(self.cfg["system_dynamics_state_idx_dict"]) + 4, self.cfg["system_dynamics_num_visualizations"], figsize=(10 * self.cfg["system_dynamics_num_visualizations"], 10))

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs = self.env.get_observations().to(self.device)
        self.real_obs_buf = torch.zeros(0, obs["policy"].shape[1], device=self.device)
        self.imagination_obs_init_buf = torch.zeros(0, obs["policy"].shape[1], device=self.device)
        self.imagination_obs_advance_buf = torch.zeros(0, obs["policy"].shape[1], device=self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            rewards_collection = torch.zeros(self.env.num_envs, self.num_steps_per_env, device=self.device)
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # process the step
                    if it >= start_iter + self.cfg["system_dynamics_warmup_iterations"]:
                        # Process env step and store in buffer
                        self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
                    self.alg.fill_history_buffer(obs)
                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        rewards_collection[:, i] = rewards / self.env.unwrapped.step_dt
                        # -- intrinsic and extrinsic rewards
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns
                self.alg.compute_returns(obs)
            
            mean_system_state_loss, mean_system_sequence_loss, mean_system_bound_loss, mean_system_kl_loss, mean_system_extension_loss, mean_system_contact_loss, mean_system_termination_loss = self.alg.update_system_dynamics()
            # update policy
            if it >= start_iter + self.cfg["system_dynamics_warmup_iterations"]:
                if self.num_imagination_envs > 0 and self.num_imagination_steps > 0:
                    real_observation, imagination_observation, per_step_reward_imagination, num_valid_imagination_envs, epistemic_uncertainty, rewbuffer_imagination, lenbuffer_imagination = self.imagine()
                    loss_dict = self.alg.update(imagination=True)
                else:
                    loss_dict = self.alg.update()
            else:
                loss_dict = {
                    "value_function": 0.0,
                    "surrogate": 0.0,
                    "entropy": 0.0,
                }
                if self.alg.rnd:
                    loss_dict["rnd"] = 0.0
                if self.alg.symmetry:
                    loss_dict["symmetry"] = 0.0


            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration + 1}.pt"))

    def imagine(self):
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.num_imagination_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.num_imagination_envs, dtype=torch.float, device=self.device)
        epistemic_uncertainty = torch.zeros(self.num_imagination_steps, device=self.device)
        state_history, action_history = self.alg.prepare_imagination()
        self.env.unwrapped.prepare_imagination()
        with torch.inference_mode():
            for i in range(self.num_imagination_steps):
                if i % self.imagination_cfg["command_resample_interval"] == 0:
                    self.env.unwrapped.sample_imagination_command()
                if self.alg.system_dynamics.architecture_config["type"] in ["rnn", "rssm"] and i > 0:
                    state_history = state_history[:, -1:]
                    action_history = action_history[:, -1:]
                imagination_obs = self.env.unwrapped.get_imagination_observation(state_history, action_history)
                imagination_actions = self.alg.act(imagination_obs)
                imagination_obs, imagination_rewards, imagination_dones, imagination_extras, state_history, action_history, uncertainty = self.env.unwrapped.imagination_step(imagination_actions, state_history, action_history)
                self.alg.process_env_step(imagination_obs, imagination_rewards, imagination_dones, imagination_extras, imagination=True)
                epistemic_uncertainty[i] = uncertainty.mean(dim=0)
                
                cur_reward_sum += imagination_rewards
                cur_episode_length += 1
                new_ids = (imagination_dones > 0).nonzero(as_tuple=False)
                rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                cur_reward_sum[new_ids] = 0
                cur_episode_length[new_ids] = 0
                
            self.alg.compute_returns(imagination_obs, imagination=True)
            
        # logs
        real_observation = self.alg.storage.observations["policy"]
        imagination_observation = torch.cat([self.alg.imagination_storage.observations["policy"], imagination_obs["policy"].unsqueeze(0)], dim=0)
        per_step_reward_imagination = self.env.unwrapped.get_imagination_reward_per_step()
        num_valid_imagination_envs = self.alg.imagination_storage.valid_env_mask.sum()
        epistemic_uncertainty = epistemic_uncertainty.mean(dim=0)
        return real_observation, imagination_observation, per_step_reward_imagination, num_valid_imagination_envs, epistemic_uncertainty, rewbuffer, lenbuffer

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs, width, pad)
        self.writer.add_scalar("Model Based/per_step_reward_sum_real", locs["rewards_collection"].mean(), locs["it"])

        if locs["it"] >= locs["start_iter"] + self.cfg["system_dynamics_warmup_iterations"]:        
            if self.num_imagination_envs > 0 and self.num_imagination_steps > 0:
                self.writer.add_scalar("Model Based/epistemic_uncertainty", locs["epistemic_uncertainty"], locs["it"])
                per_step_reward_sum_imagination = torch.stack(list(locs["per_step_reward_imagination"].values())).sum()
                self.writer.add_scalar("Model Based/num_valid_imagination_envs", locs["num_valid_imagination_envs"], locs["it"])
                self.real_obs_buf = torch.cat((self.real_obs_buf, locs["real_observation"].flatten(0, 1)), dim=0)[-self.cfg["pca_obs_buf_size"]:]
                self.imagination_obs_init_buf = torch.cat((self.imagination_obs_init_buf, locs["imagination_observation"][0]), dim=0)[-self.cfg["pca_obs_buf_size"]:]
                self.imagination_obs_advance_buf = torch.cat((self.imagination_obs_advance_buf, locs["imagination_observation"][1:].flatten(0, 1)), dim=0)[-self.cfg["pca_obs_buf_size"]:]
                self.writer.add_scalar("Model Based/per_step_reward_sum_imagination", per_step_reward_sum_imagination, locs["it"])
                if len(locs["rewbuffer_imagination"]) > 0:
                    self.writer.add_scalar("Train/mean_reward_imagination", statistics.mean(locs["rewbuffer_imagination"]), locs["it"])
                    self.writer.add_scalar("Train/mean_episode_length_imagination", statistics.mean(locs["lenbuffer_imagination"]), locs["it"])
                for term, value in locs["per_step_reward_imagination"].items():
                    self.writer.add_scalar(f"Model Based/per_step_reward_imagination/{term}", value, locs["it"])
                if locs["it"] % self.save_interval == 0:
                    self.plotter.plot_pca(
                        self.ax0,
                        [self.real_obs_buf, self.imagination_obs_init_buf, self.imagination_obs_advance_buf],
                        legend_list=["Real", "Imagination-0", "Imagination-1+"]
                        )
                    self.writer.add_figure("Model Based/obs_distribution", self.fig0, locs["it"])
        self.writer.add_scalar("System Dynamics/state_loss", locs["mean_system_state_loss"], locs["it"])
        self.writer.add_scalar("System Dynamics/sequence_loss", locs["mean_system_sequence_loss"], locs["it"])
        self.writer.add_scalar("System Dynamics/bound_loss", locs["mean_system_bound_loss"], locs["it"])
        self.writer.add_scalar("System Dynamics/kl_loss", locs["mean_system_kl_loss"], locs["it"])
        if self.system_extension_dim > 0:
            self.writer.add_scalar("System Dynamics/extension_loss", locs["mean_system_extension_loss"], locs["it"])
        if self.system_contact_dim > 0:
            self.writer.add_scalar("System Dynamics/contact_loss", locs["mean_system_contact_loss"], locs["it"])
        if self.system_termination_dim > 0:
            self.writer.add_scalar("System Dynamics/termination_loss", locs["mean_system_termination_loss"], locs["it"])
        self.writer.add_scalar("System Dynamics/learning_rate", self.alg.system_dynamics_learning_rate, locs["it"])
        
        if locs["it"] % self.save_interval == 0:
            state_traj, action_traj, extension_traj, contact_traj, termination_traj, state_traj_pred, action_traj_pred, extension_traj_pred, contact_traj_pred, termination_traj_pred, traj_autoregressive_error, traj_autoregressive_error_noised_dict = self.alg.evaluate_system_dynamics()
            state_traj = self.state_normalizer.inverse(state_traj)
            action_traj = self.action_normalizer.inverse(action_traj)
            state_traj_pred = self.state_normalizer.inverse(state_traj_pred)
            action_traj_pred = self.action_normalizer.inverse(action_traj_pred)
            self.writer.add_scalar("System Dynamics/autoregressive_error", traj_autoregressive_error, locs["it"])
            self.plotter.plot_trajectories(
                self.ax1,
                None,
                state_traj[:self.cfg["system_dynamics_num_visualizations"]],
                action_traj[:self.cfg["system_dynamics_num_visualizations"]],
                extension_traj[:self.cfg["system_dynamics_num_visualizations"]] if extension_traj is not None else None,
                contact_traj[:self.cfg["system_dynamics_num_visualizations"]] if contact_traj is not None else None,
                termination_traj[:self.cfg["system_dynamics_num_visualizations"]] if termination_traj is not None else None,
                self.cfg["system_dynamics_state_idx_dict"],
                )
            self.plotter.plot_trajectories(
                self.ax1,
                self.alg.system_dynamics.history_horizon,
                state_traj_pred[:self.cfg["system_dynamics_num_visualizations"]],
                action_traj_pred[:self.cfg["system_dynamics_num_visualizations"]],
                extension_traj_pred[:self.cfg["system_dynamics_num_visualizations"]] if extension_traj_pred is not None else None,
                contact_traj_pred[:self.cfg["system_dynamics_num_visualizations"]] if contact_traj_pred is not None else None,
                termination_traj_pred[:self.cfg["system_dynamics_num_visualizations"]] if termination_traj_pred is not None else None,
                self.cfg["system_dynamics_state_idx_dict"],
                prediction=True
                )
            self.fig1.align_ylabels()
            self.writer.add_figure("System Dynamics/trajectories", self.fig1, locs["it"])
            for noise_scale, value in traj_autoregressive_error_noised_dict.items():
                self.writer.add_scalar(f"System Dynamics/autoregressive_error_noised_{noise_scale}", value, locs["it"])

    def save(self, path: str, infos=None):
        # -- Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "system_dynamics_state_dict": self.alg.system_dynamics.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "system_dynamics_optimizer_state_dict": self.alg.system_dynamics_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # -- Save RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None):
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        # -- Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # -- Load system dynamics model
        if self.cfg["load_system_dynamics"]:
            if self.cfg["system_dynamics_load_path"] is not None:
                system_dynamics_loaded_dict = torch.load(self.cfg["system_dynamics_load_path"])
            else:
                system_dynamics_loaded_dict = loaded_dict
            self.alg.system_dynamics.load_state_dict(system_dynamics_loaded_dict["system_dynamics_state_dict"])
        # -- Load RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # -- load optimizer if used
        if load_optimizer and resumed_training:
            # -- algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # -- System dynamics
            self.alg.system_dynamics_optimizer.load_state_dict(loaded_dict["system_dynamics_optimizer_state_dict"])
            # -- RND optimizer if used
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def _construct_algorithm(self, obs) -> PPO:
        """Construct the actor-critic algorithm."""
        # resolve RND config
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # resolve symmetry config
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # initialize the actor-critic
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        
        # initialize system dynamics
        self.system_dynamics_cfg = self.cfg["system_dynamics"]
        self.imagination_cfg = self.cfg["imagination"]
        state_normalizer_cfg = self.imagination_cfg["state_normalizer"]
        action_normalizer_cfg = self.imagination_cfg["action_normalizer"]
        system_state_dim = obs["system_state"].shape[-1]
        if "system_extension" in obs:
            self.system_extension_dim = obs["system_extension"].shape[-1]
        else:
            self.system_extension_dim = 0
        if "system_contact" in obs:
            self.system_contact_dim = obs["system_contact"].shape[-1]
        else:
            self.system_contact_dim = 0
        if "system_termination" in obs:
            self.system_termination_dim = obs["system_termination"].shape[-1]
        else:
            self.system_termination_dim = 0
        system_dynamics = SystemDynamicsEnsemble(
            system_state_dim, 
            self.env.num_actions, 
            self.system_extension_dim,
            self.system_contact_dim, 
            self.system_termination_dim, 
            self.device, 
            **self.system_dynamics_cfg
        )
        self.state_normalizer = EmpiricalNormalization(shape=[system_dynamics.state_dim], until=1.0e8, eps=1.0e-8).to(self.device).eval()
        self.action_normalizer = EmpiricalNormalization(shape=[system_dynamics.action_dim], until=1.0e8, eps=1.0e-8).to(self.device).eval()
        state_normalizer_state_dict = {
            "_mean": torch.tensor(state_normalizer_cfg["mean"], device=self.device).unsqueeze(0),
            "_std": torch.tensor(state_normalizer_cfg["std"], device=self.device).unsqueeze(0),
            "_var": torch.square(torch.tensor(state_normalizer_cfg["std"], device=self.device).unsqueeze(0)),
            "count": torch.tensor(0, dtype=torch.long),
        }
        self.state_normalizer.load_state_dict(state_normalizer_state_dict)
        action_normalizer_state_dict = {
            "_mean": torch.tensor(action_normalizer_cfg["mean"], device=self.device).unsqueeze(0),
            "_std": torch.tensor(action_normalizer_cfg["std"], device=self.device).unsqueeze(0),
            "_var": torch.square(torch.tensor(action_normalizer_cfg["std"], device=self.device).unsqueeze(0)),
            "count": torch.tensor(0, dtype=torch.long),
        }
        self.action_normalizer.load_state_dict(action_normalizer_state_dict)

        alg: MBPOPPO = alg_class(actor_critic, system_dynamics, self.state_normalizer, self.action_normalizer, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        self.num_imagination_envs = self.imagination_cfg["num_envs"]
        self.num_imagination_steps = self.imagination_cfg["num_steps"]
        self.env.unwrapped.num_imagination_envs = self.num_imagination_envs
        self.env.unwrapped.num_imagination_steps = self.num_imagination_steps
        self.env.unwrapped.imagination_state_normalizer = self.state_normalizer
        self.env.unwrapped.imagination_action_normalizer = self.action_normalizer
        self.env.unwrapped.system_dynamics = alg.system_dynamics
        self.env.unwrapped.uncertainty_penalty_weight = self.imagination_cfg["uncertainty_penalty_weight"]

        # initialize the storage
        alg.init_storage(
            "rl",
            self.num_imagination_envs,
            self.num_imagination_steps,
            obs,
            [self.env.num_actions],
            imagination=True,
        )

        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg
