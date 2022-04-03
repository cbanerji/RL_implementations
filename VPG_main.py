import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
from new import *

policy_flag = None

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()

    env = gym.make('Swimmer-v2')
    if isinstance(env.observation_space, Box) == True:
        policy_flag = 0
    if isinstance(env.action_space, Discrete) == True:
        policy_flag = 1

    #train(env_name=args.env_name, render=args.render, lr=args.lr)
    trn = train("Swimmer-v2")
