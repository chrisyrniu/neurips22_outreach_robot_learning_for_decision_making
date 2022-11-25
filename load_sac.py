#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer, Batch, to_numpy
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from utils import make_mujoco_env
import gym


def load_sac(resume_path, env, device='cuda'):
    actor_lr = 0.001
    critic_lr = 0.001
    hidden_sizes = [256, 256]
    resume_path = resume_path

    env, train_envs, test_envs = make_mujoco_env(
        env, 0, 1, 1, obs_norm=False
    )

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]
    seed = 0
    # seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    # model
    net_a = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
    actor = ActorProb(
        net_a,
        action_shape,
        max_action=max_action,
        device=device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    net_c1 = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device,
    )
    net_c2 = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device,
    )
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
    )

    policy.load_state_dict(torch.load(resume_path, map_location=device))
    print("Loaded agent from: ", resume_path)

    return policy