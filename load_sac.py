#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint

import numpy as np
import torch
from mujoco_env import make_mujoco_env
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
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.device = device
    args.actor_lr = 0.001
    args.critic_lr = 0.001
    args.hidden_sizes = [256, 256]
    args.resume_path = resume_path

    env, train_envs, test_envs = make_mujoco_env(
        env, 0, 1, 1, obs_norm=False
    )

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    args.seed = 0
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # model
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net_a,
        args.action_shape,
        max_action=args.max_action,
        device=args.device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
    )

    policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
    print("Loaded agent from: ", args.resume_path)

    return policy