import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logutil
from torch.distributions import Categorical

ts = logutil.TimeSeries('Reinforcement Learning')

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')
args = parser.parse_args()


ROCK = 0
PAPER = 1
SCISSORS = 2

# An implementation of Rock-Paper-Scissors
class Environment():
    def __init__(self):
        self.our_prev_action = np.zeros(3)
        self.adv_prev_action = np.zeros(3)
        self.cumulative_actions = np.zeros((2,3), dtype=int)
        # The adversary picks a strategy: all-in rock, paper, or scissors
        # Our agent must learn to react to this strategy
        self.adversary_strategy = np.random.randint(3)

    def reset(self):
        state = np.zeros(6)
        self.our_prev_action = np.zeros(3)
        self.adv_prev_action = np.zeros(3)
        self.cumulative_actions = np.zeros((2,3), dtype=int)
        self.adversary_strategy = np.random.randint(3)
        return state

    def adversary_action(self):
        action = np.zeros(3)
        action[self.adversary_strategy] = 1.0
        return action

    def step(self, action):
        state = np.zeros(6)
        state[:3] = self.our_prev_action
        state[3:] = self.adv_prev_action

        self.our_prev_action = np.zeros(3)
        self.our_prev_action[action] = 1
        self.adv_prev_action = self.adversary_action()

        self.cumulative_actions[0] += self.our_prev_action.astype(int)
        self.cumulative_actions[1] += self.adv_prev_action.astype(int)

        reward = 0
        done = False
        info = {}
        return state, reward, done, info

    def get_reward(self, verbose=False):
        blue = self.cumulative_actions[0].astype(float)
        red = self.cumulative_actions[1].astype(float)

        if verbose:
            print('Modeling battle between {} blue units and {} enemy units'.format(
                sum(blue), sum(red)))
            print('Blue: {}'.format(blue))
            print('Red: {}'.format(red))

        while max(blue) > 0 and max(red) > 0:
            red_damage = model_battle(blue, red, verbose=False)
            blue_damage = model_battle(red, blue)
            red = apply_damage(red, red_damage)
            blue = apply_damage(blue, blue_damage)

        blue_left = blue.sum()
        red_left = red.sum()
        if verbose:
            print('Battle finished: {:.2f} blue and {:.2f} red units remain'.format(
                blue_left, red_left))
        return blue_left - red_left


# Model how much damage Blue will do to Red
def model_battle(blue, red, verbose=False):
    br, bp, bs = blue
    if verbose:
        print('Blue has: {:.02f} rocks, {:.02f} paper, {:.02f} scissors'.format(br, bp, bs))
    rr, rp, rs = red
    if verbose:
        print('Red has: {:.02f} rocks, {:.02f} paper, {:.02f} scissors'.format(rr, rp, rs))

    # All units contribute some amount of damage
    standard_dps = np.array([1, 1, 1]) * (br + bp + bs + .1)

    # Each unit that counters another applies extra damage to its adversary
    vs_rock = (rr > 0) * bp * 2.0
    vs_paper = (rp > 0) * bs * 2.0
    vs_scissors = (rs > 0) * br * 2.0
    bonus_dps = np.array([vs_rock, vs_paper, vs_scissors])

    if verbose:
        print('Blue will do {:.02f} dps vs Red'.format(total_dps))
    return (standard_dps + bonus_dps) * .01


def apply_damage(army, damage):
    # Subtract damage from all surviving units
    damage_per_unit = 3.0 / (army > 0).sum()
    army -= damage * damage_per_unit
    army = np.clip(army, 0, None)
    return army


env = Environment()

torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(6, 128)
        self.gru = nn.GRU(128, 128)
        self.affine2 = nn.Linear(128, 3)

        self.reset_gru()
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x, self.hx = self.gru(x.unsqueeze(0), self.hx)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

    def reset_gru(self):
        self.hx = torch.zeros(1, 1, 128)


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
    policy_loss = torch.cat(policy_loss).sum() * args.learning_rate
    policy_loss.backward()
    optimizer.step()
    policy.reset_gru()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        #verbose = i_episode % args.log_interval == 0
        verbose = False

        state = env.reset()
        for t in range(10):
            action = select_action(state)
            state, _, done, _ = env.step(action)
            policy.rewards.append(0)

        reward = env.get_reward(verbose)
        policy.rewards[-1] = reward
        finish_episode()

        ts.collect('reward', reward)
        ts.print_every(1)

        if verbose:
            print('Episode {}\tReward: {:.2f}'.format(i_episode, reward))



if __name__ == '__main__':
    main()
