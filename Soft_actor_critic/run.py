import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory

parser = argparse.ArgumentParser(description='Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Swimmer-v2",
                    help='Mujoco Environment name')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates current policy(default: True)')
parser.add_argument('--eval_every', type=int, default=5000,
                    help='Evaluation frequency of the mean policy')
parser.add_argument('--eval_episodes', type=int, default= 5,
                    help='Number of evaluation episodes')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α ')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=0, metavar='N',
                    help='random seed ')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of timesteps, policy optimization will run')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates after each timestep')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per timestep ')
parser.add_argument('--replay_memory_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer(default: 10^6)')
parser.add_argument('--additional_description', type=int, default='_', metavar='N',
                    help='Include additional information')
args = parser.parse_args()

# Environment
env = gym.make(args.env_name)

#Seeding enerything
env.seed(args.seed)
env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Declare SAC Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

# Setting up tensorboard, save logs to this location
writer = SummaryWriter('runs_Swimmer-v2/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.seed,args.info,"autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

def evaluate_policy(total_numsteps):
    avg_reward = 0.
    eval_episodes = args.eval_episodes

    for _  in range(eval_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

        avg_reward += episode_reward
    avg_reward /= eval_episodes

    # Writing to tensorboard
    writer.add_scalar('avg_reward/test', avg_reward, total_numsteps)

    # Print onscreen description
    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {}".format(eval_episodes, round(avg_reward, 2)))
    print("----------------------------------------")

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        # Evaluate current policy
        if total_numsteps % args.eval_every == 0 and args.eval is True:
            evaluate_policy(total_numsteps)

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it reaches time horizon.
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Add transition to replay memory buffer

        state = next_state

    if total_numsteps > args.num_steps:
        break

    # Writing to tensorboard
    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))


env.close()
