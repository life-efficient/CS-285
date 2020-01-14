import torch
from time import sleep
import gym
from utils.NN_boilerplate import NN
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

writer = SummaryWriter()


class ValueDataset(Dataset):
    def __init__(self, experiences):
        self.examples = []
        for (state, action, reward, new_state) in experiences:
            print(state)
            target = reward + discount_factor * value(new_state)
            self.examples.append(state, target)

    def __len__(self):
        return self.len(examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def train(env, value, value_table):
    for policy_idx in range(num_updates): # every update creates a new policy

        # POLICY EVALUATION step 1 = RUN/SIMULATE EPISODES 
        # experiences = []
        for state in range(env.observation_space.n):
        # for episode_idx in range(episodes_per_update):
        #     done = False
        #     state = env.reset()
        #     total_reward = 0
        #     while not done:
        #         state = torch.tensor(state)
        #         action = value_table.policy(state)
        #         print('action:', action)
        #         new_state, reward, done, _ = env.step(action)
        #         total_reward += reward
        #         env.render()
        #         sleep(0.1)
        #         experiences.append((state, action, reward, new_state))
        #         state = new_state
        #         print('STATE;',state)
        #         print(done)
        #         print()