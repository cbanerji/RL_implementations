import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box

# Define the policy network, for continuous action space
class Net(nn.Module):
    def __init__(self,state_size = 11, action_size = 2, seed=12, hidden_size=32, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(Net, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.mu.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)

    def forward(self, obs):
        x = F.relu(self.fc1(obs), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std


def train01(env_name, hidden_sizes=[32], lr=1e-2,
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    net = Net(state_size, action_size)
    # make function to compute action distribution
    def get_policy(obs):
        mu, std = net(obs)
        return Normal(loc=mu, scale=std)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().detach()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act).sum(axis=-1) #for Normal
        return -(logp * weights).mean()

    # make optimizer
    optimizer = Adam(net.parameters(), lr=lr)


    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  #act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  act=torch.tensor([item.cpu().detach().numpy() for item in batch_acts]),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32))
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens


    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

def agent(env, policy_flag):
    if policy_flag == 0:
        train01(env)
    elif  policy_flag == 1:
        print("Choose Discrete policy training")
