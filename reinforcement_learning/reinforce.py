import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


ROCK = 0
PAPER = 1
SCISSORS = 2

# An implementation of Rock-Paper-Scissors
class Environment():
    def __init__(self):
        self.our_prev_action = np.zeros(3)
        self.adv_prev_action = np.zeros(3)

    def reset(self):
        state = np.zeros(6)
        return state

    def adversary_action(self):
        action = np.zeros(3)
        action[SCISSORS] = 1.0
        return action

    def step(self, action):
        state = np.zeros(6)
        state[:3] = self.our_prev_action
        state[3:] = self.adv_prev_action

        self.our_prev_action = np.zeros(3)
        self.our_prev_action[action] = 1
        self.adv_prev_action = self.adversary_action()

        reward = self.get_reward()
        done = False
        info = {}
        return state, reward, done, info

    def get_reward(self):
        our_action = self.our_prev_action
        adv_action = self.adv_prev_action
        if our_action.argmax() == ROCK:
            if adv_action.argmax() == ROCK:
                return 0
            elif adv_action.argmax() == PAPER:
                return -1
            elif adv_action.argmax() == SCISSORS:
                return 1
        elif our_action.argmax() == PAPER:
            if adv_action.argmax() == ROCK:
                return 1
            elif adv_action.argmax() == PAPER:
                return 0
            elif adv_action.argmax() == SCISSORS:
                return -1
        elif our_action.argmax() == SCISSORS:
            if adv_action.argmax() == ROCK:
                return -1
            elif adv_action.argmax() == PAPER:
                return 1
            elif adv_action.argmax() == SCISSORS:
                return 0
        assert False


env = Environment()

torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(6, 128)
        self.affine2 = nn.Linear(128, 3)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)

        average_reward = np.mean(policy.rewards)
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tAverage reward: {:.2f}'.format(
                i_episode, average_reward))


if __name__ == '__main__':
    main()
