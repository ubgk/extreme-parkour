# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import code

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
from legged_gym.utils import webviewer


class DepthWrapper(torch.nn.Module):
    def __init__(self, depth_encoder):
        super().__init__()

        self.depth_encoder = depth_encoder.eval()

    def forward(self, depth, obs_proprio, hidden_states_in):
        obs_proprio[:, 6:8] = 0.
        depth_latent_and_yaw, hidden_states_out = self.depth_encoder(depth, obs_proprio, hidden_states_in=hidden_states_in)
        depth_latent = depth_latent_and_yaw[:, :-2]
        yaw = depth_latent_and_yaw[:, -2:]

        return depth_latent, yaw, hidden_states_out

class ActorWrapper(torch.nn.Module):
    def __init__(self, actor, estimator=None):
        super().__init__()

        self.hist_encoder = actor.history_encoder.eval()
        self.actor = actor.actor_backbone.eval()

        # base lin vel estimator
        self.estimator = estimator.eval()

    def forward(self, depth_latent, obs_proprio, obs_hist, obs_priv = None):
        assert not (obs_priv is None and self.estimator is None), \
                'You have not provided obs_priv although there is ' \
                'no velocity estimator!'

        if obs_priv is None:
            obs_priv = self.estimator(obs_proprio)

        hist_latent = self.hist_encoder(obs_hist) # obs[:, -self.num_hist*self.num_prop:]
        self.hist_latent = hist_latent

        backbone_input = torch.cat([obs_proprio, depth_latent, obs_priv, hist_latent], dim=1)
        backbone_output = self.actor(backbone_input)

        return backbone_output

class DepthActorWrapper(torch.nn.Module):
    """ Branchless wrapper module for depth encoder + actor network. """
    def __init__(self, depth_wrapper, actor_wrapper, estimator=None):
        super().__init__()

        self.depth_wrapper = depth_wrapper
        self.actor_wrapper = actor_wrapper

    def forward(self, depth, depth_latent, yaw, update_depth, obs_proprio, obs_hist, hidden_states_in, obs_priv = None):
        new_depth_latent, new_yaw, hidden_states_out = self.depth_wrapper(depth, obs_proprio, hidden_states_in)

        hidden_states_out = (update_depth * hidden_states_out) + (1 - update_depth) * hidden_states_in
        depth_latent = (update_depth * new_depth_latent) + (1 - update_depth) * depth_latent

        yaw = (update_depth * new_yaw) + (1 - update_depth) * yaw
        obs_proprio[:, 6:8] = 1.5 * yaw

        actions = self.actor_wrapper(depth_latent, obs_proprio, obs_hist, obs_priv)

        return actions, hidden_states_out, depth_latent, yaw

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

def play(args):
    if args.web:
        web_viewer = webviewer.WebViewer()
    faulthandler.enable()
    exptid = args.exptid
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 16 if not args.save else 64
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {"smooth slope": 0.,
                                    "rough slope up": 0.0,
                                    "rough slope down": 0.0,
                                    "rough stairs up": 0.,
                                    "rough stairs down": 0.,
                                    "discrete": 0.,
                                    "stepping stones": 0.0,
                                    "gaps": 0.,
                                    "smooth flat": 0,
                                    "pit": 0.0,
                                    "wall": 0.0,
                                    "platform": 0.,
                                    "large stairs up": 0.,
                                    "large stairs down": 0.,
                                    "parkour": 0.2,
                                    "parkour_hurdle": 0.2,
                                    "parkour_flat": 0.,
                                    "parkour_step": 0.2,
                                    "parkour_gap": 0.2,
                                    "demo": 0.2}

    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True

    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    depth_latent_buffer = []
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)

    policy = ppo_runner.get_inference_policy(device=env.device)
    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    if env.cfg.depth.use_camera:
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None


    depth_wrapper = DepthWrapper(depth_encoder)
    actor_wrapper = ActorWrapper(ppo_runner.alg.depth_actor,
                                 estimator=ppo_runner.alg.estimator)

    depth_actor_wrapper = DepthActorWrapper(depth_wrapper, actor_wrapper)
    hidden_states = torch.zeros((1, env.num_envs, 512), device=env.device)

    depth_latent = torch.zeros((env.num_envs, 32), device=env.device)
    yaw = torch.zeros((env.num_envs, 2), device=env.device)

    for i in range(10*int(env.max_episode_length)):
        # Branchless depth actor wrapper
        obs_proprio = obs[:, :env.cfg.env.n_proprio].clone()
        obs_hist = obs[:, -env.cfg.env.history_len*env.cfg.env.n_proprio:].clone()

        if infos["depth"] is not None:
            depth_buf = infos["depth"].clone()
            update_depth = 1.0
        else:
            update_depth = 0.0

        branchless_actions, hidden_states, depth_latent, yaw = depth_actor_wrapper(depth_buf, depth_latent, yaw, update_depth, obs_proprio, obs_hist, hidden_states)

        # Original depth actor
        if env.cfg.depth.use_camera:
            if infos["depth"] is not None:
                obs_student = obs[:, :env.cfg.env.n_proprio].clone()
                obs_student[:, 6:8] = 0
                depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
                depth_latent = depth_latent_and_yaw[:, :-2]
                yaw = depth_latent_and_yaw[:, -2:]
            obs[:, 6:8] = 1.5*yaw

        else:
            depth_latent = None

        priv_explicit = estimator(obs[:, :env.cfg.env.n_proprio])
        actor = ppo_runner.alg.depth_actor
        offset = actor.num_prop + actor.num_scan
        obs[:, offset:offset + actor.num_priv_explicit] = priv_explicit

        if hasattr(ppo_runner.alg, "depth_actor"):
            actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
        else:
            actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)

        torch.testing.assert_allclose(actions, branchless_actions, rtol=1e-04, atol=1e-04)
        print("Max action diff:", torch.max(torch.abs(actions - branchless_actions)).item())
        # torch.onnx.export(actor_wrapper, (depth_latent_and_yaw[:1], obs_proprio[:1], obs_hist[:1]), 'relaxed_actor.onnx', input_names=['depth_latent_and_yaw', 'obs_proprio', 'obs_hist'], output_names=['actions'])

        obs, _, rews, dones, infos = env.step(actions.detach())
        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
        print("time:", env.episode_length_buf[env.lookat_id].item() / 50,
              "cmd vx", env.commands[env.lookat_id, 0].item(),
              "actual vx", env.base_lin_vel[env.lookat_id, 0].item(), )

        id = env.lookat_id


if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
