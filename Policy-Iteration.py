import torch
import gym
from utils.NN_boilerplate import NN
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

env = gym.make('CartPole-v0')

value = NN([np.prod(env.observation_space.shape), 16, 16, 1], distribution=False)

def train():
    for policy_idx in range(num_updates): # every update creates a new policy
        # RUN EPISODES
        for episode_idx in range(episodes_per_update):
            done = False
            state = env.reset()
            while not done:
                state = torch.tensor(state)
                                
                for a in range(env.action_space.n): # cant resample action spaces because you'd have to revisit them

                    advantage = reward
                    print(value(state))
                # action = # argmax over actions
                k


episodes_per_update = 10
num_updates = 100
train()
